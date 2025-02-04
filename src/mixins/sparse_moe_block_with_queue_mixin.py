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

    def _process_expert_queue(self, expert_idx, received_sequences: List):
        queue = self.queues[expert_idx]
        if queue.is_empty() and not received_sequences:
            return []

        sequences_to_process = received_sequences + queue.dequeue_all()

        states = torch.stack([seq.cached_hidden_state for seq in sequences_to_process])
        expert_output = self.experts[expert_idx](states)

        for seq, output in zip(sequences_to_process, expert_output):
            seq.expert_outputs_cache[expert_idx] = output

        return sequences_to_process


    def process_decode(self, hidden_states, expert_mask, selected_expert_indices, running_batch, hidden_dim):
        # Check if any sequence in the running_batch is high priority
        global_high_priority = any(seq.priority == 1 for seq in running_batch.sequences)
        
        sequences_to_process = []
        final_states = []
        final_sequences = []
        
        if global_high_priority or not self.scheduler.has_high_priority_request():
            # Iterate over all experts if any high priority sequence exists
            for expert_idx in range(self.num_experts):
                top_x, current_state = self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
                token_indices = top_x.tolist()
                selected_sequences = [running_batch.sequences[idx] for idx in token_indices] if token_indices else []
                for seq, state in zip(selected_sequences, current_state):
                    seq.cached_hidden_state = state
                # Dequeue all items from the expert's queue and include them in the processing
                selected_sequences.extend(self.queues[expert_idx].dequeue_all())
                processed_sequences = self._process_expert_queue(expert_idx, selected_sequences)
                sequences_to_process.extend(processed_sequences)
                for seq in processed_sequences:
                    if len(seq.expert_outputs_cache) == self.top_k:
                        output = sum(seq.expert_outputs_cache.values())
                        final_states.append(output)
                        seq.expert_outputs_cache.clear()
                        final_sequences.append(seq)
        else:
            # Simply enqueue all sequences when no high priority is found
            expert_inputs = {expert_idx: self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
                             for expert_idx in selected_expert_indices}
            for expert_idx, (top_x, current_state) in expert_inputs.items():
                token_indices = top_x.tolist()
                selected_sequences = [running_batch.sequences[idx] for idx in token_indices]
                for seq, state in zip(selected_sequences, current_state):
                    seq.cached_hidden_state = state
                self.queues[expert_idx].enqueue_many(selected_sequences)
        
        running_batch.clear()
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
