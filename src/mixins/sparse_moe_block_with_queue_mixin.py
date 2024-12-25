import torch
import time
from src.queues.fcfs_queue import FCFSQueue


class SparseMoeBlockWithQueuesMixin:
    def __init__(self, num_experts, *args, **kwargs):
        self.queues = [FCFSQueue() for _ in range(num_experts)]
        self.num_experts = num_experts  # Ensure num_experts is stored

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
        threshold = 16  # TODO: Define your threshold here. Should be adjusted either dynamically or from a config file
        time_limit = 0.1  # TODO: Define your time limit in seconds here. Should be adjusted either dynamically or from a config file

        if queue.is_empty():
            return []

        head_item, head_timestamp = queue.peek()
        current_time = time.time()
        queue_size = queue.size()
        
        if current_time - head_timestamp >= time_limit or queue_size >= threshold:
            num_to_process = queue_size if current_time - head_timestamp >= time_limit else threshold
            expert_sequences = [queue.dequeue() for _ in range(num_to_process)]
            states = [seq.cached_hidden_state for seq in expert_sequences]

            if states:
                batched_states = torch.stack(states)
                expert_output = self.experts[expert_idx](batched_states)
                
                for seq, output in zip(expert_sequences, expert_output):
                    seq.expert_outputs_cache[expert_idx] = output

            return expert_sequences

        return []

    def _aggregate_final_states(self, sequences_list, running_batch, hidden_dim, hidden_states):
        final_states = []
        for seq in sequences_list:
            if len(seq.expert_outputs_cache) == self.top_k:
                output = sum(seq.expert_outputs_cache.values())
                final_states.append(output)
                seq.expert_outputs_cache.clear()
                running_batch.add_sequence(seq)

        return torch.cat(final_states) if final_states else torch.zeros(
            (0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

    def process_decode(self, hidden_states, expert_mask, selected_expert_indices, running_batch, hidden_dim):
        # Enqueue sequences to expert queues
        for expert_idx in selected_expert_indices:
            top_x, current_state = self._get_expert_inputs(hidden_states, expert_mask, expert_idx)
            token_indices = top_x.tolist()
            selected_sequences = [running_batch.sequences[idx] for idx in token_indices]
            
            for seq, state in zip(selected_sequences, current_state):
                seq.cached_hidden_state = state
                self.queues[expert_idx].enqueue(seq)
        
        running_batch.clear()
        
        # Process queues and collect sequences
        sequences_list = []
        for expert_idx in range(self.num_experts):
            sequences_list.extend(self._process_expert_queue(expert_idx))

        return self._aggregate_final_states(sequences_list, running_batch, hidden_dim, hidden_states)