"""QueueAwareMoEWrapper: Drop-in replacement for HF MoE blocks with per-expert queuing.

This module wraps an existing HuggingFace MoE block (e.g., MixtralSparseMoeBlock)
via composition — no subclassing of HF classes. It:
- Preserves the original forward() signature: hidden_states → (hidden_states, router_logits)
- In PREFILL mode: passes through to the original block (unchanged)
- In DECODE mode: routes tokens to per-expert FIFO queues for scheduling
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.queues.fcfs_queue import FCFSQueue

logger = logging.getLogger(__name__)


class QueueAwareMoEWrapper(nn.Module):
    """Wraps a HuggingFace MoE block to add per-expert queue scheduling.

    The wrapper uses the original block's gate and experts — it does NOT
    copy or re-implement them. This means any HF optimization applied to the
    original block (quantization, torch.compile, etc.) is preserved.
    """

    def __init__(self, original_moe_block: nn.Module, scheduler=None):
        super().__init__()
        # Store reference to original block (composition, not inheritance)
        self.original = original_moe_block

        # Reference the original's sub-modules directly
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

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_decode_mode(self, decode: bool):
        self._decode_mode = decode

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass with same signature as original MoE block.

        Returns:
            Tuple of (final_hidden_states, router_logits)
        """
        if not self._decode_mode:
            # PREFILL: pass through to original (full parallel processing)
            return self.original(hidden_states)
        else:
            # DECODE: per-expert queue routing
            return self._decode_with_queues(hidden_states)

    def _decode_with_queues(self, hidden_states: torch.Tensor) -> tuple:
        """Decode with per-expert queue routing."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing using original gate
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Build expert mask: [num_experts, top_k, num_tokens]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]

        # Process each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = hidden_states_flat[top_x]
            current_routing_weights = routing_weights[top_x, idx].unsqueeze(-1)

            # Process through the original expert module
            expert_output = self.experts[expert_idx](current_state)
            expert_output *= current_routing_weights

            # Accumulate results
            final_hidden_states.index_add_(0, top_x, expert_output.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits

    def _decode_with_queues_preemptive(self, hidden_states: torch.Tensor,
                                        running_batch=None) -> tuple:
        """Decode with per-expert queues and preemption support.

        This is the full QLLM scheduling path — used when a scheduler is set
        and preemptive scheduling is desired. Tokens are queued per-expert,
        and high-priority requests can preempt best-effort execution.

        This method requires running_batch to access sequence objects for
        state caching and priority checking.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        # Check if preemption is needed
        should_preempt = (
            self.scheduler is not None
            and running_batch is not None
            and not any(seq.priority == 1 for seq in running_batch.sequences)
            and self.scheduler.has_high_priority_request()
        )

        if should_preempt:
            # Enqueue all tokens and return early — let scheduler handle LS requests
            self._enqueue_tokens(hidden_states_flat, expert_mask, running_batch)
            running_batch.clear()
            return (
                torch.zeros((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device),
                router_logits,
            )

        # Normal decode: enqueue, process queues, aggregate
        self._enqueue_tokens(hidden_states_flat, expert_mask, running_batch)
        running_batch.clear()

        final_states = []
        final_sequences = []
        for expert_idx in range(self.num_experts):
            processed = self._process_expert_queue(expert_idx)
            for seq in processed:
                if len(seq.expert_outputs_cache) == self.top_k:
                    output = sum(seq.expert_outputs_cache.values())
                    final_states.append(output)
                    seq.expert_outputs_cache.clear()
                    final_sequences.append(seq)

        if running_batch is not None:
            running_batch.add_sequence(final_sequences)

        if final_states:
            result = torch.cat(final_states).reshape(-1, hidden_dim)
        else:
            result = torch.zeros((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

        return result, router_logits

    def _enqueue_tokens(self, hidden_states_flat, expert_mask, running_batch):
        """Enqueue tokens to per-expert queues."""
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            token_indices = top_x.tolist()
            current_states = hidden_states_flat[top_x]
            for i, (seq_idx, state) in enumerate(zip(token_indices, current_states)):
                seq = running_batch.sequences[seq_idx]
                seq.cached_hidden_state = state
                self.queues[expert_idx].enqueue(seq)

    def _process_expert_queue(self, expert_idx):
        """Process all tokens in an expert's queue."""
        queue = self.queues[expert_idx]
        if queue.is_empty():
            return []
        sequences = []
        states = []
        while not queue.is_empty():
            seq = queue.dequeue()
            sequences.append(seq)
            states.append(seq.cached_hidden_state)
        states = torch.stack(states)
        expert_output = self.experts[expert_idx](states)
        for seq, output in zip(sequences, expert_output):
            seq.expert_outputs_cache[expert_idx] = output
        return sequences

    # --- Queue inspection methods ---

    def count_queued_tokens(self) -> int:
        return sum(queue.size() for queue in self.queues)

    def has_queued_items(self) -> bool:
        return any(not queue.is_empty() for queue in self.queues)
