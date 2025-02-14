from typing import List
import torch
from src.queues.fcfs_queue import FCFSQueue

class SparseMoeBlockWithQueuesMixin:
    def __init__(self, num_experts, *args, **kwargs):
        self.queues = [FCFSQueue() for _ in range(num_experts)]
        self.num_experts = num_experts  # Ensure num_experts is stored
        self.scheduler = None
        
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _get_expert_inputs(self, hidden_states, expert_mask, expert_idx):
        idx, top_x = torch.where(expert_mask[expert_idx])
        return top_x, hidden_states[top_x]

    def process_prefill(self, hidden_states, expert_mask, selected_expert_indices, final_hidden_states):
        for expert_idx in selected_expert_indices:
            top_x, current_state = self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
            current_hidden_states = self.experts[expert_idx](current_state)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states

    def _process_expert_queue(self, expert_idx):
        queue = self.queues[expert_idx]
        if queue.size() < 1:
            return []
        sequences_to_process = []
        states = []
        while not queue.is_empty():
            seq = queue.dequeue()
            sequences_to_process.append(seq)
            states.append(seq.cached_hidden_state)
        states = torch.stack(states)
        expert_output = self.experts[expert_idx](states)
        for seq, output in zip(sequences_to_process, expert_output):
            seq.expert_outputs_cache[expert_idx] = output
        return sequences_to_process

    def process_decode(self, hidden_states, expert_mask, selected_expert_indices, running_batch, hidden_dim):
        # Enqueue sequences based on hidden state extraction
        if any(seq.priority == 1 for seq in running_batch.sequences) or not self.scheduler.has_high_priority_request():
            for expert_idx in range(self.num_experts):
                top_x, current_state = self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
                token_indices = top_x.tolist()
                selected_sequences = [running_batch.sequences[idx] for idx in token_indices] if token_indices else []
                for seq, state in zip(selected_sequences, current_state):
                    seq.cached_hidden_state = state
                    self.queues[expert_idx].enqueue(seq)
        else:
            # NEW: if scheduler has high priority and no high priority sequence, enqueue and return immediately.
            expert_inputs = {expert_idx: self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
                             for expert_idx in selected_expert_indices}
            for expert_idx, (top_x, current_state) in expert_inputs.items():
                token_indices = top_x.tolist()
                selected_sequences = [running_batch.sequences[idx] for idx in token_indices]
                for seq, state in zip(selected_sequences, current_state):
                    seq.cached_hidden_state = state
                    self.queues[expert_idx].enqueue(seq)
            running_batch.clear()
            return torch.zeros(
                (0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
                    
        running_batch.clear()

        final_states = []
        final_sequences = []
        for expert_idx in range(self.num_experts):
            processed_sequences = self._process_expert_queue(expert_idx)
            for seq in processed_sequences:
                if len(seq.expert_outputs_cache) == self.top_k:
                    output = sum(seq.expert_outputs_cache.values())
                    final_states.append(output)
                    seq.expert_outputs_cache.clear()
                    final_sequences.append(seq)
        running_batch.add_sequence(final_sequences)
        return torch.cat(final_states) if final_states else torch.zeros(
            (0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

    def count_queued_tokens(self):
        # Returns the sum of tokens queued in all expert queues.
        return sum(queue.size() for queue in self.queues)
    
    def has_queued_items(self):
        # Returns True if any of the expert queues is not empty.
        return any(not queue.is_empty() for queue in self.queues)
