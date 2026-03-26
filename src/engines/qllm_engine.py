"""QllmEngine: Unified engine with layer-by-layer decode and per-expert queuing.

Supports two decode modes:
- Standard: full model.forward() pass (fast, no queuing)
- Queued: layer-by-layer processing with threshold-based expert queuing.
  Attention is computed per-sequence (correct positions/KV cache).
  MoE blocks batch tokens with threshold-based firing.
  Deferred tokens are tracked per-layer and resumed when their expert
  queues fill from subsequent iterations' tokens.
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import DynamicCache

from src.engines.base_engine import BaseEngine
from src.models.model_adapter import ModelAdapter
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_processor import SamplingProcessor
from src.samplers.sampling_params import SamplingParams
from src.sequence.stage import Stage

logger = logging.getLogger(__name__)


class QllmEngine(BaseEngine):

    def __init__(
        self,
        model_adapter: ModelAdapter,
        cache_manager: Optional[SequenceCacheManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        use_queues: bool = False,
        queue_threshold: int = 1,
    ):
        self.model_adapter = model_adapter
        self.model = model_adapter.model
        self.cache_manager = cache_manager or SequenceCacheManager()
        self.sampling_params = sampling_params or SamplingParams()
        self.sampling_processor = SamplingProcessor(self.sampling_params)
        self.use_queues = use_queues
        self.queue_threshold = queue_threshold

        # Build layer→wrapper mapping
        self._wrapper_map: Dict[int, object] = {}
        if model_adapter.moe_wrappers:
            for i, layer in enumerate(model_adapter.model.model.layers):
                for name in ["block_sparse_moe", "sparse_moe", "moe"]:
                    block = getattr(layer, name, None)
                    if block is not None and block in model_adapter.moe_wrappers:
                        self._wrapper_map[i] = block
                        break

        # Cross-iteration deferred token tracking:
        # layer_idx → set of seq_ids waiting in that layer's expert queues
        self._deferred_at_layer: Dict[int, Set[int]] = defaultdict(set)
        self._deferred_sequences: Dict[int, object] = {}

    # ------------------------------------------------------------------ #
    def run_batch(self, batch) -> "Batch":
        sequences = batch.sequences
        is_decode = batch.is_decode()
        if is_decode and self.use_queues and self._wrapper_map:
            return self._run_queued_decode(batch)
        return self._run_standard(batch)

    # ------------------------------------------------------------------ #
    #  Standard forward                                                    #
    # ------------------------------------------------------------------ #
    def _run_standard(self, batch) -> "Batch":
        sequences = batch.sequences
        is_decode = batch.is_decode()
        self.model_adapter.set_decode_mode(is_decode)
        for w in self.model_adapter.moe_wrappers:
            w.set_batch_context(sequences=None, use_queues=False)

        input_ids, attention_mask = self._prepare_inputs(batch)
        past_key_values = self.cache_manager.assemble_batch_cache(sequences)

        with torch.inference_mode():
            outputs = self.model_adapter.forward(
                input_ids=input_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True,
            )

        kv_caches = self.cache_manager.split_to_sequences(
            outputs.past_key_values, sequences
        )
        self.sampling_processor.sample_batch(outputs.logits, sequences, kv_caches)
        return batch

    # ------------------------------------------------------------------ #
    #  Layer-by-layer queued decode                                        #
    # ------------------------------------------------------------------ #
    def _run_queued_decode(self, batch) -> "Batch":
        """Layer-by-layer decode with cross-iteration deferral.

        At each layer:
        1. Run attention per-sequence (new tokens only — deferred tokens
           already had attention at their deferred layer)
        2. Run MoE with threshold-based queuing. New tokens are enqueued
           which may trigger expert fires that also process old deferred tokens.
        3. Completed tokens (current + previously-deferred) proceed to next layer.
        4. Incomplete tokens stay in expert queues (deferred tracking updated).
        """
        sequences = batch.sequences
        base = self.model_adapter.model.model
        device = next(base.parameters()).device
        self.model_adapter.set_decode_mode(True)

        # Per-sequence KV caches (load for ALL sequences including deferred)
        per_seq_caches = {}
        for seq in sequences:
            per_seq_caches[seq.sequence_id] = self.cache_manager.get_or_create(
                seq.sequence_id
            )
        for sid in self._deferred_sequences:
            if sid not in per_seq_caches:
                per_seq_caches[sid] = self.cache_manager.get_or_create(sid)

        # Separate fresh from deferred
        fresh_seqs = [
            seq for seq in sequences
            if seq.sequence_id not in self._deferred_sequences
        ]

        if not fresh_seqs:
            # All deferred — force-fire expert queues to prevent starvation
            self._force_fire_deferred(sequences, per_seq_caches, base, device)
            return batch

        # Embed last tokens for fresh sequences only
        input_ids = self._get_decode_tokens(fresh_seqs, device)
        all_hidden = base.embed_tokens(input_ids)

        active_seqs = list(fresh_seqs)
        active_hidden = all_hidden

        with torch.inference_mode():
            for layer_idx, layer in enumerate(base.layers):
                if not active_seqs:
                    break

                wrapper = self._wrapper_map.get(layer_idx)

                # --- Attention: per-sequence for active (new) tokens ---
                attn_outputs = []
                for j, seq in enumerate(active_seqs):
                    h = active_hidden[j:j + 1]
                    cache = per_seq_caches.get(seq.sequence_id)
                    if cache is None:
                        cache = self.cache_manager.get_or_create(seq.sequence_id)
                        per_seq_caches[seq.sequence_id] = cache

                    past_len = cache.get_seq_length() if cache.key_cache else 0
                    cache_pos = torch.tensor([past_len], device=device, dtype=torch.long)
                    pos_ids = cache_pos.unsqueeze(0)
                    pos_emb = base.rotary_emb(h, pos_ids)

                    residual = h
                    normed = layer.input_layernorm(h)
                    attn_out, _ = layer.self_attn(
                        hidden_states=normed,
                        position_embeddings=pos_emb,
                        attention_mask=None,
                        past_key_value=cache,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    attn_outputs.append(residual + attn_out)

                active_hidden = torch.cat(attn_outputs, dim=0)

                # --- MoE with threshold-based queuing ---
                if wrapper is not None:
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)

                    # Save post-attention state for each active token
                    for j, seq in enumerate(active_seqs):
                        seq._deferred_residual = residual[j:j + 1]
                        seq._deferred_layer = layer_idx

                    completed_h, completed_seqs, newly_deferred = \
                        self._moe_with_threshold(wrapper, normed, active_seqs, layer_idx)

                    # Track newly deferred sequences
                    for seq in newly_deferred:
                        self._deferred_at_layer[layer_idx].add(seq.sequence_id)
                        self._deferred_sequences[seq.sequence_id] = seq

                    if completed_seqs:
                        active_hidden = torch.cat(completed_h, dim=0)
                        active_seqs = completed_seqs
                    else:
                        active_hidden = torch.zeros(
                            0, 1, normed.shape[-1], device=device, dtype=normed.dtype
                        )
                        active_seqs = []
                else:
                    # Standard FFN
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)
                    ffn = getattr(layer, 'block_sparse_moe',
                                  getattr(layer, 'mlp', None))
                    if ffn is not None:
                        out = ffn(normed)
                        if isinstance(out, tuple):
                            out = out[0]
                        active_hidden = residual + out

        # --- Final: norm + lm_head + sampling ---
        if active_seqs:
            active_hidden = base.norm(active_hidden)
            logits = self.model_adapter.model.lm_head(active_hidden)
            kv_list = [per_seq_caches[seq.sequence_id] for seq in active_seqs]
            self.sampling_processor.sample_batch(logits, active_seqs, kv_list)

        # Save ALL caches (active + deferred)
        for sid, cache in per_seq_caches.items():
            self.cache_manager.per_sequence_caches[sid] = cache

        return batch

    def _force_fire_deferred(self, sequences, per_seq_caches, base, device):
        """Force-fire all expert queues to unblock deferred sequences.

        Called when all sequences in the batch are deferred (starvation).
        Processes deferred tokens through remaining layers with threshold=1.
        """
        # Find all layers with deferred tokens
        for layer_idx in sorted(self._deferred_at_layer.keys()):
            wrapper = self._wrapper_map.get(layer_idx)
            if wrapper is None:
                continue

            # Force-fire ALL expert queues at this layer (threshold=1)
            for expert_idx in range(wrapper.num_experts):
                if not wrapper.queues[expert_idx].is_empty():
                    wrapper._process_expert_queue(expert_idx)

            # Collect completed tokens
            deferred_ids = self._deferred_at_layer.get(layer_idx, set())
            completed_seqs = []
            completed_h = []

            for seq_id in list(deferred_ids):
                seq = self._deferred_sequences.get(seq_id)
                if seq is None:
                    deferred_ids.discard(seq_id)
                    continue
                cache = getattr(seq, 'expert_outputs_cache', {})
                if len(cache) >= wrapper.top_k:
                    expert_out = sum(cache.values())
                    h = seq._deferred_residual + expert_out.unsqueeze(0).unsqueeze(0)
                    completed_seqs.append(seq)
                    completed_h.append(h)
                    cache.clear()
                    seq._cached_expert_weights = {}
                    deferred_ids.discard(seq_id)
                    self._deferred_sequences.pop(seq_id, None)

            if not completed_seqs:
                continue

            # Process completed tokens through remaining layers
            active_hidden = torch.cat(completed_h, dim=0)
            active_seqs = completed_seqs
            layer_list = list(base.layers)

            with torch.inference_mode():
                for li in range(layer_idx + 1, len(layer_list)):
                    if not active_seqs:
                        break
                    layer = layer_list[li]
                    next_wrapper = self._wrapper_map.get(li)

                    # Attention per-sequence
                    attn_outputs = []
                    for j, seq in enumerate(active_seqs):
                        h_s = active_hidden[j:j + 1]
                        c = per_seq_caches.get(seq.sequence_id)
                        if c is None:
                            c = self.cache_manager.get_or_create(seq.sequence_id)
                            per_seq_caches[seq.sequence_id] = c
                        past_len = c.get_seq_length() if c.key_cache else 0
                        cp = torch.tensor([past_len], device=device, dtype=torch.long)
                        pe = base.rotary_emb(h_s, cp.unsqueeze(0))
                        residual = h_s
                        normed = layer.input_layernorm(h_s)
                        ao, _ = layer.self_attn(
                            hidden_states=normed, position_embeddings=pe,
                            attention_mask=None, past_key_value=c,
                            use_cache=True, cache_position=cp,
                        )
                        attn_outputs.append(residual + ao)
                    active_hidden = torch.cat(attn_outputs, dim=0)

                    # MoE (force threshold=1 for remaining layers)
                    if next_wrapper is not None:
                        residual = active_hidden
                        normed = layer.post_attention_layernorm(active_hidden)
                        for j, seq in enumerate(active_seqs):
                            seq._deferred_residual = residual[j:j + 1]
                        ch, cs, _ = self._moe_with_threshold(
                            next_wrapper, normed, active_seqs, li,
                            force_threshold=1,
                        )
                        if cs:
                            active_hidden = torch.cat(ch, dim=0)
                            active_seqs = cs
                        else:
                            active_seqs = []
                    else:
                        residual = active_hidden
                        normed = layer.post_attention_layernorm(active_hidden)
                        ffn = getattr(layer, 'block_sparse_moe',
                                      getattr(layer, 'mlp', None))
                        if ffn:
                            out = ffn(normed)
                            if isinstance(out, tuple):
                                out = out[0]
                            active_hidden = residual + out

            # Final norm + lm_head + sampling
            if active_seqs:
                active_hidden = base.norm(active_hidden)
                logits = self.model_adapter.model.lm_head(active_hidden)
                for seq in active_seqs:
                    self.cache_manager.per_sequence_caches[seq.sequence_id] = \
                        per_seq_caches[seq.sequence_id]
                kv_list = [per_seq_caches[s.sequence_id] for s in active_seqs]
                self.sampling_processor.sample_batch(logits, active_seqs, kv_list)

    def _moe_with_threshold(self, wrapper, normed_hidden, active_seqs, layer_idx,
                            force_threshold=None):
        """MoE with threshold. Checks both current and deferred tokens for completion.

        Returns (completed_hidden, completed_seqs, deferred_seqs).
        """
        hidden_dim = normed_hidden.shape[-1]
        hidden_flat = normed_hidden.view(-1, hidden_dim)

        # Route current tokens
        router_logits = wrapper.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, wrapper.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_flat.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=wrapper.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        # Enqueue current tokens
        for expert_idx in range(wrapper.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            tok_indices = top_x.tolist()
            states = hidden_flat[top_x]
            weights = routing_weights[top_x, idx]
            for i, ti in enumerate(tok_indices):
                seq = active_seqs[ti]
                seq.cached_hidden_state = states[i]
                if not hasattr(seq, '_cached_expert_weights'):
                    seq._cached_expert_weights = {}
                seq._cached_expert_weights[expert_idx] = weights[i]
                seq._token_position = ti
                wrapper.queues[expert_idx].enqueue(seq)

        # Fire experts above threshold
        threshold = force_threshold if force_threshold is not None else self.queue_threshold
        for expert_idx in range(wrapper.num_experts):
            if wrapper.queues[expert_idx].size() >= threshold:
                wrapper._process_expert_queue(expert_idx)

        # Collect completions from BOTH current and previously-deferred tokens
        all_candidates = list(active_seqs)
        deferred_ids = self._deferred_at_layer.get(layer_idx, set())
        for seq_id in list(deferred_ids):
            seq = self._deferred_sequences.get(seq_id)
            if seq is not None and seq not in all_candidates:
                all_candidates.append(seq)

        completed_h = []
        completed_seqs = []
        deferred_seqs = []

        for seq in all_candidates:
            cache = getattr(seq, 'expert_outputs_cache', {})
            if len(cache) == wrapper.top_k:
                expert_out = sum(cache.values())
                h = seq._deferred_residual + expert_out.unsqueeze(0).unsqueeze(0)
                completed_h.append(h)
                completed_seqs.append(seq)
                cache.clear()
                seq._cached_expert_weights = {}
                # Clear deferred tracking
                deferred_ids.discard(seq.sequence_id)
                self._deferred_sequences.pop(seq.sequence_id, None)
            else:
                deferred_seqs.append(seq)

        # Sort completed by token position
        if completed_seqs:
            paired = sorted(
                zip(completed_seqs, completed_h),
                key=lambda x: getattr(x[0], '_token_position', 0)
            )
            completed_seqs, completed_h = zip(*paired)
            completed_seqs = list(completed_seqs)
            completed_h = list(completed_h)

        return completed_h, completed_seqs, deferred_seqs

    # ------------------------------------------------------------------ #
    #  Input helpers                                                       #
    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, batch):
        sequences = batch.sequences
        is_decode = batch.is_decode()
        if is_decode:
            ids = [
                seq.generated_tokens[-1:] if seq.generated_tokens.numel() > 0
                else seq.input_ids[-1:]
                for seq in sequences
            ]
        else:
            ids = [seq.input_ids for seq in sequences]
        padded = pad_sequence(ids, batch_first=True, padding_value=0)
        if is_decode:
            max_len = max(
                (seq.get_total_sequence_length() for seq in sequences), default=1
            )
            mask = torch.ones(len(sequences), max_len, dtype=torch.long, device=padded.device)
        else:
            mask = (padded != 0).long()
        return padded, mask

    def _get_decode_tokens(self, sequences, device):
        tokens = []
        for seq in sequences:
            if seq.generated_tokens.numel() > 0:
                tokens.append(seq.generated_tokens[-1:])
            else:
                tokens.append(seq.input_ids[-1:])
        return torch.stack(tokens).to(device)
