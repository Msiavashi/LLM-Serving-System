"""QllmEngine: Two paths — standard model.forward() and layer-by-layer queued decode.

Standard path: full model.forward() with batched attention. Fast, HF-native.
Queued path: iterates through layers. Batched attention using assembled cache.
At MoE blocks, tokens go through per-expert queues. Experts fire when their
queue reaches the threshold. Completed tokens proceed, deferred tokens stay
in queues across iterations. This is the original QLLM mechanism.
"""
import logging
from collections import defaultdict
from typing import Dict, Optional, Set

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import DynamicCache

from src.engines.base_engine import BaseEngine
from src.models.model_adapter import ModelAdapter
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_processor import SamplingProcessor
from src.samplers.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class QllmEngine(BaseEngine):

    def __init__(self, model_adapter: ModelAdapter,
                 cache_manager: Optional[SequenceCacheManager] = None,
                 sampling_params: Optional[SamplingParams] = None,
                 use_queues: bool = False, queue_threshold: int = 1):
        self.model_adapter = model_adapter
        self.model = model_adapter.model
        self.cache_manager = cache_manager or SequenceCacheManager()
        self.sampling_params = sampling_params or SamplingParams()
        self.sampling_processor = SamplingProcessor(self.sampling_params)
        self.use_queues = use_queues
        self.queue_threshold = queue_threshold

        # Map layer_idx → wrapper for layers that have MoE blocks
        self._wrapper_map: Dict[int, object] = {}
        if model_adapter.moe_wrappers:
            for i, layer in enumerate(model_adapter.model.model.layers):
                for attr in ["block_sparse_moe", "sparse_moe", "moe"]:
                    blk = getattr(layer, attr, None)
                    if blk is not None and blk in model_adapter.moe_wrappers:
                        self._wrapper_map[i] = blk
                        break

        # Deferred sequences: seq_id → (seq_object, layer_idx)
        self._deferred: Dict[int, tuple] = {}

    # ------------------------------------------------------------------ #
    def run_batch(self, batch):
        is_decode = batch.is_decode()
        if is_decode and self.use_queues and self._wrapper_map:
            return self._queued_decode(batch)
        return self._standard_forward(batch)

    # ------------------------------------------------------------------ #
    # Standard: one model.forward() call                                   #
    # ------------------------------------------------------------------ #
    def _standard_forward(self, batch):
        seqs = batch.sequences
        self.model_adapter.set_decode_mode(batch.is_decode())
        for w in self.model_adapter.moe_wrappers:
            w.set_batch_context(None, False)

        ids, mask = self._prepare_inputs(batch)
        past = self.cache_manager.assemble_batch_cache(seqs)
        with torch.inference_mode():
            out = self.model_adapter.forward(ids, mask, past, use_cache=True)
        kvs = self.cache_manager.split_to_sequences(out.past_key_values, seqs)
        self.sampling_processor.sample_batch(out.logits, seqs, kvs)
        return batch

    # ------------------------------------------------------------------ #
    # Queued decode: layer-by-layer with per-expert queues                 #
    # ------------------------------------------------------------------ #
    def _queued_decode(self, batch):
        base = self.model_adapter.model.model
        device = next(base.parameters()).device
        all_seqs = batch.sequences

        # Separate fresh vs deferred sequences
        fresh = [s for s in all_seqs if s.sequence_id not in self._deferred]
        if not fresh:
            # Starvation: force-fire all queues with threshold=1
            self._force_fire(all_seqs, base, device)
            return batch

        self.model_adapter.set_decode_mode(True)

        # Embed fresh tokens
        tok_ids = self._get_last_tokens(fresh, device)
        hidden = base.embed_tokens(tok_ids)  # [B, 1, D]

        # Running batch: sequences currently flowing through layers
        running = list(fresh)
        running_hidden = hidden

        with torch.inference_mode():
            for li, layer in enumerate(base.layers):
                if not running:
                    break

                wrapper = self._wrapper_map.get(li)

                # --- Check if deferred tokens at this layer can resume ---
                if wrapper:
                    resumed, resumed_h = self._try_resume(li, wrapper)
                    if resumed:
                        running.extend(resumed)
                        running_hidden = torch.cat(
                            [running_hidden] + resumed_h, dim=0
                        )

                # --- Attention (batched, using assembled cache) ---
                past = self.cache_manager.assemble_batch_cache(running)
                if past is None:
                    past = DynamicCache()
                seq_len = max((s.get_total_sequence_length() for s in running), default=1)
                cache_pos = torch.tensor([seq_len - 1], device=device, dtype=torch.long)
                pos_ids = cache_pos.unsqueeze(0)
                pos_emb = base.rotary_emb(running_hidden, pos_ids)
                causal_mask = base._update_causal_mask(None, running_hidden, cache_pos, past, False)

                residual = running_hidden
                normed = layer.input_layernorm(running_hidden)
                attn_out, _ = layer.self_attn(
                    hidden_states=normed, position_embeddings=pos_emb,
                    attention_mask=causal_mask, past_key_value=past,
                    use_cache=True, cache_position=cache_pos,
                )
                running_hidden = residual + attn_out

                # Save KV back to per-sequence caches
                self.cache_manager.disassemble_batch_cache(past, running)

                # --- MoE with per-expert queuing ---
                if wrapper:
                    residual = running_hidden
                    normed = layer.post_attention_layernorm(running_hidden)

                    # Save state for deferral
                    for j, s in enumerate(running):
                        s._deferred_residual = residual[j:j+1]
                        s._deferred_layer = li

                    completed_h, completed_s, deferred_s = self._moe_queued(
                        wrapper, normed, running, li
                    )

                    # Track deferred
                    for s in deferred_s:
                        self._deferred[s.sequence_id] = (s, li)

                    running = completed_s
                    running_hidden = torch.cat(completed_h, dim=0) if completed_h else \
                        torch.zeros(0, 1, normed.shape[-1], device=device, dtype=normed.dtype)
                else:
                    # Standard FFN
                    residual = running_hidden
                    normed = layer.post_attention_layernorm(running_hidden)
                    ffn = getattr(layer, 'block_sparse_moe', getattr(layer, 'mlp', None))
                    if ffn:
                        out = ffn(normed)
                        if isinstance(out, tuple): out = out[0]
                        running_hidden = residual + out

        # Final: norm → lm_head → sample
        if running:
            running_hidden = base.norm(running_hidden)
            logits = self.model_adapter.model.lm_head(running_hidden)
            kvs = [self.cache_manager.per_sequence_caches.get(s.sequence_id, DynamicCache())
                   for s in running]
            self.sampling_processor.sample_batch(logits, running, kvs)

        return batch

    def _moe_queued(self, wrapper, normed, running, layer_idx):
        """Route through per-expert queues. Experts fire at threshold."""
        hidden_dim = normed.shape[-1]
        flat = normed.view(-1, hidden_dim)

        router_logits = wrapper.gate(flat)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, wrapper.top_k, dim=-1)
        rw /= rw.sum(dim=-1, keepdim=True)
        rw = rw.to(flat.dtype)

        mask = F.one_hot(sel, num_classes=wrapper.num_experts).permute(2, 1, 0)

        # Enqueue
        for ei in range(wrapper.num_experts):
            idx, top_x = torch.where(mask[ei])
            if top_x.numel() == 0: continue
            for i, ti in enumerate(top_x.tolist()):
                seq = running[ti]
                seq.cached_hidden_state = flat[top_x[i]]
                if not hasattr(seq, '_cached_expert_weights'):
                    seq._cached_expert_weights = {}
                seq._cached_expert_weights[ei] = rw[top_x[i], idx[i]]
                wrapper.queues[ei].enqueue(seq)

        # Fire experts above threshold
        for ei in range(wrapper.num_experts):
            if wrapper.queues[ei].size() >= self.queue_threshold:
                wrapper._process_expert_queue(ei)

        # Partition into completed vs deferred
        completed_h, completed_s, deferred_s = [], [], []
        # Check running sequences AND previously deferred at this layer
        candidates = list(running)
        for sid, (seq, li) in list(self._deferred.items()):
            if li == layer_idx and seq not in candidates:
                candidates.append(seq)

        for seq in candidates:
            ec = getattr(seq, 'expert_outputs_cache', {})
            if len(ec) == wrapper.top_k:
                h = seq._deferred_residual + sum(ec.values()).unsqueeze(0).unsqueeze(0)
                completed_h.append(h)
                completed_s.append(seq)
                ec.clear()
                seq._cached_expert_weights = {}
                self._deferred.pop(seq.sequence_id, None)
            else:
                deferred_s.append(seq)

        return completed_h, completed_s, deferred_s

    def _try_resume(self, layer_idx, wrapper):
        """Check if deferred tokens at this layer completed (expert fired previously)."""
        resumed, resumed_h = [], []
        for sid, (seq, li) in list(self._deferred.items()):
            if li != layer_idx: continue
            ec = getattr(seq, 'expert_outputs_cache', {})
            if len(ec) >= wrapper.top_k:
                h = seq._deferred_residual + sum(ec.values()).unsqueeze(0).unsqueeze(0)
                resumed.append(seq)
                resumed_h.append(h)
                ec.clear()
                seq._cached_expert_weights = {}
                del self._deferred[sid]
        return resumed, resumed_h

    def _force_fire(self, seqs, base, device):
        """Force-fire all expert queues to unblock starvation."""
        for li, wrapper in self._wrapper_map.items():
            for ei in range(wrapper.num_experts):
                if not wrapper.queues[ei].is_empty():
                    wrapper._process_expert_queue(ei)
            # Collect completed
            completed_h, completed_s = [], []
            for sid, (seq, sli) in list(self._deferred.items()):
                if sli != li: continue
                ec = getattr(seq, 'expert_outputs_cache', {})
                if len(ec) >= wrapper.top_k:
                    h = seq._deferred_residual + sum(ec.values()).unsqueeze(0).unsqueeze(0)
                    completed_h.append(h)
                    completed_s.append(seq)
                    ec.clear()
                    del self._deferred[sid]

            if not completed_s: continue

            # Process remaining layers for completed tokens
            running = completed_s
            running_h = torch.cat(completed_h, dim=0)
            layers = list(base.layers)
            for li2 in range(li + 1, len(layers)):
                if not running: break
                layer = layers[li2]
                w2 = self._wrapper_map.get(li2)

                past = self.cache_manager.assemble_batch_cache(running)
                if past is None: past = DynamicCache()
                sl = max((s.get_total_sequence_length() for s in running), default=1)
                cp = torch.tensor([sl - 1], device=device, dtype=torch.long)
                pe = base.rotary_emb(running_h, cp.unsqueeze(0))
                cm = base._update_causal_mask(None, running_h, cp, past, False)

                res = running_h
                n = layer.input_layernorm(running_h)
                ao, _ = layer.self_attn(hidden_states=n, position_embeddings=pe,
                                        attention_mask=cm, past_key_value=past,
                                        use_cache=True, cache_position=cp)
                running_h = res + ao
                self.cache_manager.disassemble_batch_cache(past, running)

                if w2:
                    res = running_h
                    n = layer.post_attention_layernorm(running_h)
                    for j, s in enumerate(running):
                        s._deferred_residual = res[j:j+1]
                    ch, cs, ds = self._moe_queued(w2, n, running, li2)
                    for s in ds: self._deferred[s.sequence_id] = (s, li2)
                    running = cs
                    running_h = torch.cat(ch, dim=0) if ch else torch.zeros(0,1,n.shape[-1],device=device,dtype=n.dtype)
                else:
                    res = running_h
                    n = layer.post_attention_layernorm(running_h)
                    ffn = getattr(layer, 'block_sparse_moe', getattr(layer, 'mlp', None))
                    if ffn:
                        o = ffn(n)
                        if isinstance(o, tuple): o = o[0]
                        running_h = res + o

            if running:
                running_h = base.norm(running_h)
                logits = self.model_adapter.model.lm_head(running_h)
                kvs = [self.cache_manager.per_sequence_caches.get(s.sequence_id, DynamicCache()) for s in running]
                self.sampling_processor.sample_batch(logits, running, kvs)

    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, batch):
        seqs = batch.sequences
        if batch.is_decode():
            ids = [s.generated_tokens[-1:] if s.generated_tokens.numel() > 0 else s.input_ids[-1:] for s in seqs]
        else:
            ids = [s.input_ids for s in seqs]
        padded = pad_sequence(ids, batch_first=True, padding_value=0)
        if batch.is_decode():
            ml = max((s.get_total_sequence_length() for s in seqs), default=1)
            mask = torch.ones(len(seqs), ml, dtype=torch.long, device=padded.device)
        else:
            mask = (padded != 0).long()
        return padded, mask

    def _get_last_tokens(self, seqs, device):
        return torch.stack([
            s.generated_tokens[-1:] if s.generated_tokens.numel() > 0 else s.input_ids[-1:]
            for s in seqs
        ]).to(device)
