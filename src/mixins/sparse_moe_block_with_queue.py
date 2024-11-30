import torch
from src.queues.fcfs_queue import FCFSQueue


class SparseMoeBlockWithQueuesMixin:
    def __init__(self, num_experts, *args, **kwargs):
        self.queues = [FCFSQueue() for _ in range(num_experts)]
        
    def process_prefill(self, hidden_states, expert_mask, selected_expert_indices, final_hidden_states):
        # Process non-decode mode more efficiently
        for expert_idx in selected_expert_indices:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[top_x]
            current_hidden_states = self.experts[expert_idx](current_state)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states
        
    def process_decode(self, hidden_states, expert_mask, selected_expert_indices, running_batch, hidden_dim):
        for expert_idx in selected_expert_indices:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[top_x]
            token_indices = top_x.tolist()
            selected_sequences = [running_batch.sequences[idx] for idx in token_indices]
            
            # Batch sequence updates
            for seq, state in zip(selected_sequences, current_state):
                seq.cached_hidden_state = state
                self.queues[expert_idx].enqueue(seq)
        
        running_batch.clear()
        
        sequences_list = []

        for expert_idx in range(self.num_experts):
            queue = self.queues[expert_idx]
            if queue.size() >= 1:
                
                expert_sequences = []
                states = []
                
                while queue.size() > 0:
                    seq = queue.dequeue()
                    states.append(seq.cached_hidden_state)
                    expert_sequences.append(seq)
                
                if states:
                    batched_states = torch.stack(states)
                    expert_output = self.experts[expert_idx](batched_states)
                    
                    for seq, output in zip(expert_sequences, expert_output):
                        seq.expert_outputs_cache[expert_idx] = output
                            
                    sequences_list.extend(expert_sequences)

        # Aggregate final states for completed sequences
        final_states = []
        for seq in sequences_list:
            if len(seq.expert_outputs_cache) == self.top_k:
                output = sum(seq.expert_outputs_cache.values())
                final_states.append(output)
                seq.expert_outputs_cache.clear()
                running_batch.add_sequence(seq)
        
        final_hidden_states = torch.cat(final_states) if final_states else torch.zeros(
            (0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        return final_hidden_states