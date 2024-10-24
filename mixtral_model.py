from typing import List
import torch
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from torch.nn import functional as F
from sequence_queue import FCFSQueue

from sequence import Sequence

class MyMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.top_k = 2
        self.queues = [FCFSQueue() for _ in range(self.num_experts)]
        self.running: List[Sequence] = []  # List of sequences currently being processed
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Iterate only over the experts that are selected in expert_mask
        selected_expert_indices = torch.where(expert_mask.sum(dim=(1, 2)) > 0)[0]

        for expert_idx in selected_expert_indices:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class MyCustomMixtral(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(config.num_hidden_layers):
            self.model.layers[i].block_sparse_moe = MyMixtralSparseMoeBlock(config)
                
    def forward(self, sequences: List[Sequence], **kwargs):
        input_ids_list = []
        attention_mask_list = []
        past_key_values_list = []
        for sequence in sequences:
            if sequence.stage == "prefill":
                
                # Pad the sequence to the model's maximum length in batch
                max_length = max([seq.input_ids.size(0) for seq in sequences])
                padding_length = max_length - sequence.input_ids.size(0)
                if padding_length > 0:
                    sequence.input_ids = F.pad(sequence.input_ids, (0, padding_length), value=self.config.pad_token_id)
                    sequence.attention_mask = F.pad(sequence.attention_mask, (0, padding_length), value=0)
                
                
                input_ids_list.append(sequence.input_ids.unsqueeze(0))
                attention_mask_list.append(sequence.attention_mask.unsqueeze(0))
            else:
                input_ids_list.append(sequence.generated_tokens[-1:].unsqueeze(0))  
                attention_mask_list.append(sequence.attention_mask[-1:].unsqueeze(0))
                past_key_values_list.append(sequence.kv_cache)

        input_ids = torch.cat(input_ids_list, dim=0)  # Shape: [batch_size, seq_len]
        attention_mask = torch.cat(attention_mask_list, dim=0)  # Shape: [batch_size, seq_len]

        if past_key_values_list:
            # Restructure past_key_values to handle each sequence's KV cache properly
            num_layers = len(self.model.layers)
            past_key_values = []
            for layer_idx in range(num_layers):
                key_states_list = []
                value_states_list = []
                for seq_kv_cache in past_key_values_list:
                    # Extract the key and value for each layer
                    key_i, value_i = seq_kv_cache[layer_idx]
                    key_states_list.append(key_i)
                    value_states_list.append(value_i)
                # Stack along batch dimension for this layer
                key_states = torch.cat(key_states_list, dim=0)
                value_states = torch.cat(value_states_list, dim=0)
                past_key_values.append((key_states, value_states))
            past_key_values = tuple(past_key_values)

            outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        else:
            outputs = super().forward(input_ids, attention_mask, **kwargs)
        
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        
        for i, sequence in enumerate(sequences):
            last_token_logits = logits[i, -1, :]
            next_token_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            
            # Extract and store the new KV cache only for the i-th sequence
            sequence_kv_cache = []
            for layer_idx, (key_layer, value_layer) in enumerate(kv_cache):
                # Extract the i-th sequence's cache for each layer
                key_i = key_layer[i].unsqueeze(0)  # Keep the batch dimension
                value_i = value_layer[i].unsqueeze(0)
                sequence_kv_cache.append((key_i, value_i))
            
            sequence.update(next_token_ids, sequence_kv_cache)
            if sequence.stage == "prefill":
                sequence.stage = "decode"

        return sequences
