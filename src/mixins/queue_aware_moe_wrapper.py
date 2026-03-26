"""QueueAwareMoEWrapper: Drop-in replacement for HF MoE blocks with per-expert queuing.

This module wraps an existing HuggingFace MoE block (e.g., MixtralSparseMoeBlock)
via composition — no subclassing of HF classes. It:
- Preserves the original forward() signature: hidden_states → (hidden_states, router_logits)
- In PREFILL mode: passes through to the original block (unchanged)
- In DECODE mode with queues: routes tokens to per-expert FIFO queues for scheduling
- In DECODE mode without queues: processes experts directly (standard MoE)
"""
import logging
from typing import List, Optional

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

    Context injection pattern: before each forward pass, the engine calls
    set_batch_context() to provide sequence-level state that the wrapper
    needs for queue management and preemption decisions.
    """

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

        self.queues = [FCFSQueue() for _ in range(self.num_experts)]
        self.scheduler = scheduler
        self._decode_mode = False
        self._use_queues = False  # Whether to use per-expert queuing in decode
        self._batch_sequences = None  # Set by engine before forward pass

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_decode_mode(self, decode: bool):
        self._decode_mode = decode

    def set_batch_context(self, sequences: Optional[list] = None, use_queues: bool = False):
        """Inject batch context before forward pass.

        Args:
            sequences: List of Sequence objects for queue management
            use_queues: If True, use per-expert queuing (QLLM preemptive path)
        """
        self._batch_sequences = sequences
        self._use_queues = use_queues

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass — same signature as original MoE block.

        Returns:
            Tuple of (final_hidden_states, router_logits)
        """
        if not self._decode_mode:
            return self.original(hidden_states)

        if self._use_queues and self._batch_sequences is not None:
            return self._decode_with_preemptive_queues(hidden_states)
        else:
            return self._decode_standard(hidden_states)

    def _decode_standard(self, hidden_states: torch.Tensor) -> tuple:
        """Standard decode: process experts directly (no queuing)."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states_flat[top_x]
            current_weights = routing_weights[top_x, idx].unsqueeze(-1)
            expert_output = self.experts[expert_idx](current_state)
            expert_output *= current_weights
            final_hidden_states.index_add_(0, top_x, expert_output.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits

    def _decode_with_preemptive_queues(self, hidden_states: torch.Tensor) -> tuple:
        """Decode with per-expert queues and preemption support.

        This is the core QLLM scheduling path:
        1. Compute routing via gate
        2. Check if preemption is needed (LS request waiting, current batch is BE)
        3. Enqueue tokens to per-expert queues
        4. Process expert queues
        5. Aggregate results for sequences that have all top-k experts complete
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        sequences = self._batch_sequences

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        # Check preemption: if current batch is all BE and LS request is waiting
        should_preempt = (
            self.scheduler is not None
            and sequences is not None
            and not any(seq.priority == 1 for seq in sequences)
            and self.scheduler.has_high_priority_request()
        )

        # Enqueue tokens to per-expert queues
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            token_indices = top_x.tolist()
            current_states = hidden_states_flat[top_x]
            current_weights = routing_weights[top_x, idx]
            for i, tok_idx in enumerate(token_indices):
                if tok_idx < len(sequences):
                    seq = sequences[tok_idx]
                    seq.cached_hidden_state = current_states[i]
                    seq.cached_routing_weight = current_weights[i]
                    seq.cached_expert_idx = expert_idx
                    self.queues[expert_idx].enqueue(seq)

        if should_preempt:
            # Return empty — scheduler will handle LS requests next
            return (
                torch.zeros((0, seq_len, hidden_dim), dtype=hidden_states.dtype,
                            device=hidden_states.device),
                router_logits,
            )

        # Process expert queues and aggregate
        final_states = []
        final_sequences = []
        for expert_idx in range(self.num_experts):
            processed = self._process_expert_queue(expert_idx)
            for seq in processed:
                if len(getattr(seq, 'expert_outputs_cache', {})) == self.top_k:
                    # All top-k experts have processed this token — aggregate
                    output = sum(seq.expert_outputs_cache.values())
                    final_states.append(output)
                    seq.expert_outputs_cache.clear()
                    final_sequences.append(seq)

        # Update batch context with completed sequences
        self._batch_sequences = final_sequences

        if final_states:
            result = torch.stack(final_states).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            result = torch.zeros((0, seq_len, hidden_dim), dtype=hidden_states.dtype,
                                 device=hidden_states.device)

        return result, router_logits

    def _process_expert_queue(self, expert_idx: int) -> list:
        """Process all tokens in an expert's queue through the expert network."""
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
            weights.append(getattr(seq, 'cached_routing_weight', torch.tensor(1.0)))
        states = torch.stack(states)
        weights = torch.stack(weights).unsqueeze(-1)
        expert_output = self.experts[expert_idx](states) * weights
        for seq, output in zip(sequences, expert_output):
            if not hasattr(seq, 'expert_outputs_cache'):
                seq.expert_outputs_cache = {}
            seq.expert_outputs_cache[expert_idx] = output
        return sequences

    def count_queued_tokens(self) -> int:
        return sum(queue.size() for queue in self.queues)

    def has_queued_items(self) -> bool:
        return any(not queue.is_empty() for queue in self.queues)
