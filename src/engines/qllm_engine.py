"""QllmEngine: Unified engine with layer-by-layer decode and per-expert queuing.

Supports two decode modes:
- Standard: full model.forward() pass (fast, no queuing)
- Queued: layer-by-layer processing with threshold-based expert queuing.
  Attention is computed per-sequence (correct positions/KV cache),
  MoE is batched across sequences (expert-level token batching).
  This is the core QLLM mechanism for preemptive scheduling.
"""
import logging
from typing import List, Optional, Tuple

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
    """Engine with HF-native forward pass and optional layer-by-layer queued decode."""

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

        # Build layer→wrapper mapping for queued decode
        self._wrapper_map = {}
        if model_adapter.moe_wrappers:
            base_layers = list(model_adapter.model.model.layers)
            for i, layer in enumerate(base_layers):
                for name in ["block_sparse_moe", "sparse_moe", "moe"]:
                    block = getattr(layer, name, None)
                    if block is not None and block in model_adapter.moe_wrappers:
                        self._wrapper_map[i] = block
                        break

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def run_batch(self, batch) -> "Batch":
        sequences = batch.sequences
        is_decode = batch.is_decode()

        if is_decode and self.use_queues and self._wrapper_map:
            return self._run_queued_decode(batch)
        else:
            return self._run_standard(batch)

    # ------------------------------------------------------------------ #
    #  Standard forward (prefill or non-queued decode)                     #
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
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
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
        """Layer-by-layer decode: attention per-sequence, MoE batched with thresholds.

        For each decoder layer:
        1. Run attention INDEPENDENTLY per sequence (correct position/KV cache)
        2. Combine hidden states into a batch for the MoE block
        3. Route tokens to per-expert queues, fire experts above threshold
        4. Only tokens with all top-k experts done proceed to the next layer
        5. Deferred tokens stay in expert queues for future iterations
        """
        sequences = batch.sequences
        base = self.model_adapter.model.model
        device = next(base.parameters()).device

        self.model_adapter.set_decode_mode(True)

        # Get per-sequence caches (NOT assembled — we run attention per-sequence)
        per_seq_caches = {}
        for seq in sequences:
            per_seq_caches[seq.sequence_id] = self.cache_manager.get_or_create(
                seq.sequence_id
            )

        # Embed last tokens: [B, 1, D]
        input_ids = self._get_decode_tokens(sequences, device)
        all_hidden = base.embed_tokens(input_ids)

        active_seqs = list(sequences)
        active_hidden = all_hidden  # [B, 1, D]

        with torch.inference_mode():
            for layer_idx, layer in enumerate(base.layers):
                if not active_seqs:
                    break

                # --- 1. Attention: per-sequence (each has own KV cache/position) ---
                attn_outputs = []
                for j, seq in enumerate(active_seqs):
                    h = active_hidden[j: j + 1]  # [1, 1, D]
                    cache = per_seq_caches[seq.sequence_id]

                    # Compute position for this sequence
                    past_len = cache.get_seq_length() if cache.key_cache else 0
                    cache_pos = torch.tensor([past_len], device=device, dtype=torch.long)
                    pos_ids = cache_pos.unsqueeze(0)  # [1, 1]
                    pos_emb = base.rotary_emb(h, pos_ids)

                    # Attention with per-sequence cache
                    residual = h
                    normed = layer.input_layernorm(h)
                    attn_out, _ = layer.self_attn(
                        hidden_states=normed,
                        position_embeddings=pos_emb,
                        attention_mask=None,  # Causal by default for single token
                        past_key_value=cache,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    h = residual + attn_out
                    attn_outputs.append(h)

                # Stack attention outputs: [B_active, 1, D]
                active_hidden = torch.cat(attn_outputs, dim=0)

                # --- 2. MoE block with threshold-based queuing ---
                wrapper = self._wrapper_map.get(layer_idx)

                if wrapper is not None:
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)

                    # Save post-attention state on each sequence (for residual on completion)
                    for j, seq in enumerate(active_seqs):
                        seq._deferred_residual = residual[j: j + 1]

                    completed_h, completed_seqs = self._moe_with_threshold(
                        wrapper, normed, active_seqs
                    )

                    if completed_seqs:
                        active_hidden = torch.cat(completed_h, dim=0)
                        active_seqs = completed_seqs
                    else:
                        active_hidden = torch.zeros(
                            0, 1, normed.shape[-1], device=device, dtype=normed.dtype
                        )
                        active_seqs = []
                else:
                    # Non-MoE layer or no wrapper — standard FFN
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)
                    ffn = getattr(layer, 'block_sparse_moe',
                                  getattr(layer, 'mlp', None))
                    if ffn is not None:
                        out = ffn(normed)
                        if isinstance(out, tuple):
                            out = out[0]
                        active_hidden = residual + out
                    else:
                        active_hidden = residual + normed

        # --- Final: norm + lm_head + sampling ---
        if active_seqs:
            active_hidden = base.norm(active_hidden)
            logits = self.model_adapter.model.lm_head(active_hidden)

            # Update per-sequence caches in the manager
            for seq in active_seqs:
                self.cache_manager.per_sequence_caches[seq.sequence_id] = \
                    per_seq_caches[seq.sequence_id]

            kv_list = [per_seq_caches[seq.sequence_id] for seq in active_seqs]
            self.sampling_processor.sample_batch(logits, active_seqs, kv_list)
        else:
            # All tokens deferred — save cache state
            for seq in sequences:
                self.cache_manager.per_sequence_caches[seq.sequence_id] = \
                    per_seq_caches[seq.sequence_id]

        return batch

    def _moe_with_threshold(self, wrapper, normed_hidden, active_seqs):
        """MoE with threshold-based expert queuing.

        Returns (completed_hidden, completed_seqs) where completed tokens
        have all top-k experts done and residual applied.
        """
        hidden_dim = normed_hidden.shape[-1]
        hidden_flat = normed_hidden.view(-1, hidden_dim)

        # Route tokens via gate
        router_logits = wrapper.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, wrapper.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_flat.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=wrapper.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [E, K, N]

        # Enqueue tokens to per-expert queues
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

        # Fire experts that meet threshold
        for expert_idx in range(wrapper.num_experts):
            if wrapper.queues[expert_idx].size() >= self.queue_threshold:
                wrapper._process_expert_queue(expert_idx)

        # Collect completed tokens (all top-k experts done)
        completed_h = []
        completed_seqs = []

        for j, seq in enumerate(active_seqs):
            cache = getattr(seq, 'expert_outputs_cache', {})
            if len(cache) == wrapper.top_k:
                expert_out = sum(cache.values())
                h = seq._deferred_residual + expert_out.unsqueeze(0).unsqueeze(0)
                completed_h.append(h)
                completed_seqs.append(seq)
                cache.clear()
                seq._cached_expert_weights = {}

        return completed_h, completed_seqs

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
            mask = torch.ones(
                len(sequences), max_len, dtype=torch.long, device=padded.device
            )
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
