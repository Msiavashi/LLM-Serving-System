import torch
from transformers import PhimoeForCausalLM
from src.batching.batch import Batch
from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
from transformers.models.phimoe.modeling_phimoe import PhimoeSparseMoeBlock, sparsemixer, PhimoeModel

class MyPhimoeSparseMoeBlock(PhimoeSparseMoeBlock):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = sparsemixer(
            router_logits,
            jitter_eps=self.router_jitter_noise,
            training=self.training,
        )
        
        # Optimize selected_experts assignment
        num_tokens = selected_experts.shape[0]
        expert_indices = torch.arange(num_tokens, device=selected_experts.device)
        selected_experts[:, 0] = expert_indices % self.num_experts
        selected_experts[:, 1] = (expert_indices + 1) % self.num_experts
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
 
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue

            # Print expert ID and number of tokens it processes
            # print(f"Expert ID: {expert_idx}, Number of tokens: {top_x.shape[0]}")

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


class PhiMoe(PhimoeForCausalLM):
    
          
    def __init__(self, config):
        super().__init__(config)
        self.model = PhimoeModel(config)
        for i in range(config.num_hidden_layers):
            self.model.layers[i].block_sparse_moe = MyPhimoeSparseMoeBlock(config)

    
    def forward(self, batch: Batch, **kwargs):
        # Set running sequences
        self.running_sequences = batch.sequences
        
        # Extract model inputs
        input_ids, attention_mask, past_key_values = self._prepare_inputs(batch)
        
        # Forward pass
        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        
        # Process outputs
        return self._process_outputs(outputs, len(self.running_sequences))
    
    def _prepare_inputs(self, batch: Batch):
        input_ids_list, attention_mask_list, past_key_values_list = batch.model_inputs.get_all_inputs()
        
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        if past_key_values_list:
            past_key_values = DynamicCache.merge_kv_caches(past_key_values_list)
        else:
            past_key_values = DynamicCache()
        
        return input_ids, attention_mask, past_key_values

    def _process_outputs(self, outputs, num_sequences):
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        
        # Split kv_cache and create a new batch
        split_kv_cache = kv_cache.split_kv_cache(num_sequences)
        new_batch = Batch(self.running_sequences)
        new_batch.update_sequences(logits, split_kv_cache)
        
        return new_batch
