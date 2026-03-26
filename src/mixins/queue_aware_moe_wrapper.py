"""QueueAwareMoEWrapper: Drop-in MoE block replacement with per-expert queuing.

Wraps any HuggingFace MoE block via composition. Preserves the original
forward() signature: hidden_states → (hidden_states, router_logits).

Three modes:
- PREFILL: pass-through to original block
- DECODE (standard): process experts with routing weights (same as original)
- DECODE (queued): per-expert FIFO queues with priority preemption
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.queues.fcfs_queue import FCFSQueue

logger = logging.getLogger(__name__)


class QueueAwareMoEWrapper(nn.Module):

    def __init__(self, original_moe_block: nn.Module, scheduler=None):
        super().__init__()
        self.original = original_moe_block
        self.gate = original_moe_block.gate
        self.experts = original_moe_block.experts
        self.num_experts = getattr(original_moe_block, "num_experts",
                                   getattr(original_moe_block, "num_local_experts",
                                           len(original_moe_block.experts)))
        self.top_k = getattr(original_moe_block, "top_k",
                             getattr(original_moe_block, "num_experts_per_tok", 2))

        # Per-expert FIFO queues
        self.queues = [FCFSQueue() for _ in range(self.num_experts)]
        self.scheduler = scheduler
        self._decode_mode = False
        self._use_queues = False
        self._batch_sequences = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_decode_mode(self, decode: bool):
        self._decode_mode = decode

    def set_batch_context(self, sequences=None, use_queues=False):
        self._batch_sequences = sequences
        self._use_queues = use_queues

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        if not self._decode_mode:
            return self.original(hidden_states)

        if self._use_queues and self._batch_sequences is not None:
            return self._queued_decode(hidden_states)
        else:
            return self._standard_decode(hidden_states)

    # ------------------------------------------------------------------ #
    #  Standard decode (same as original, no queuing)                      #
    # ------------------------------------------------------------------ #

    def _standard_decode(self, hidden_states: torch.Tensor) -> tuple:
        """Decode without queuing — functionally identical to original block."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_flat.dtype)

        final_hidden = torch.zeros_like(hidden_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_flat[top_x]
            current_weights = routing_weights[top_x, idx].unsqueeze(-1)
            expert_output = self.experts[expert_idx](current_state) * current_weights
            final_hidden.index_add_(0, top_x, expert_output.to(hidden_flat.dtype))

        return final_hidden.reshape(batch_size, seq_len, hidden_dim), router_logits

    # ------------------------------------------------------------------ #
    #  Queued decode with per-expert FIFO queues + preemption               #
    # ------------------------------------------------------------------ #

    def _queued_decode(self, hidden_states: torch.Tensor) -> tuple:
        """Decode with per-expert queuing and priority preemption.

        This is the core QLLM mechanism:
        1. Compute routing via gate
        2. Check for priority preemption (LS waiting while processing BE)
        3. Route tokens through per-expert FIFO queues
        4. Process expert queues
        5. Aggregate results for completed tokens
        6. Return full-batch result (zeros for preempted tokens)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        sequences = self._batch_sequences

        # Gate routing
        router_logits = self.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_flat.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        # --- Priority preemption check ---
        should_preempt = (
            self.scheduler is not None
            and sequences is not None
            and not any(seq.priority == 1 for seq in sequences)
            and self.scheduler.has_high_priority_request()
        )

        if should_preempt:
            # Return zeros — current BE batch gets no MoE contribution.
            # The decoder layer's residual connection gives: residual + 0 = residual.
            # The scheduler will process the LS request in the next iteration.
            logger.debug("Preempting BE batch — LS request waiting")
            return (
                torch.zeros(batch_size, seq_len, hidden_dim,
                            dtype=hidden_states.dtype, device=hidden_states.device),
                router_logits,
            )

        # --- Enqueue tokens to per-expert queues ---
        final_hidden = torch.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            tok_indices = top_x.tolist()
            current_states = hidden_flat[top_x]
            current_weights = routing_weights[top_x, idx]

            # Enqueue with per-expert weight tracking
            for i, ti in enumerate(tok_indices):
                if ti < len(sequences):
                    seq = sequences[ti]
                    seq.cached_hidden_state = current_states[i]
                    if not hasattr(seq, '_cached_expert_weights'):
                        seq._cached_expert_weights = {}
                    seq._cached_expert_weights[expert_idx] = current_weights[i]
                    self.queues[expert_idx].enqueue(seq)

        # --- Process all expert queues ---
        for expert_idx in range(self.num_experts):
            if not self.queues[expert_idx].is_empty():
                self._process_expert_queue(expert_idx)

        # --- Aggregate completed tokens ---
        for ti, seq in enumerate(sequences):
            cache = getattr(seq, 'expert_outputs_cache', {})
            if len(cache) == self.top_k:
                output = sum(cache.values())
                final_hidden[ti] = output
                cache.clear()
                seq._cached_expert_weights = {}

        return final_hidden.reshape(batch_size, seq_len, hidden_dim), router_logits

    def _process_expert_queue(self, expert_idx: int) -> list:
        """Process all tokens in an expert's queue."""
        queue = self.queues[expert_idx]
        if queue.is_empty():
            return []
        sequences = []
        states = []
        weights = []
        while not queue.is_empty():
            seq = queue.dequeue()
            sequences.append(seq)
            states.append(seq.cached_hidden_state)
            w = seq._cached_expert_weights.get(
                expert_idx,
                torch.tensor(1.0, device=seq.cached_hidden_state.device)
            )
            weights.append(w)
        states = torch.stack(states)
        weights = torch.stack(weights).unsqueeze(-1)
        expert_output = self.experts[expert_idx](states) * weights
        for seq, output in zip(sequences, expert_output):
            if not hasattr(seq, 'expert_outputs_cache'):
                seq.expert_outputs_cache = {}
            seq.expert_outputs_cache[expert_idx] = output
        return sequences

    # ------------------------------------------------------------------ #
    #  Queue inspection                                                    #
    # ------------------------------------------------------------------ #

    def count_queued_tokens(self) -> int:
        return sum(queue.size() for queue in self.queues)

    def has_queued_items(self) -> bool:
        return any(not queue.is_empty() for queue in self.queues)
