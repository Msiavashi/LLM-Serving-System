from typing import List
import torch
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralDecoderLayer, MixtralConfig, MixtralModel, MoeModelOutputWithPast
from torch.nn import functional as F
from src.queues import FCFSQueue
from src.sequence import Sequence
from src.batching.batch import Batch
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.cache.dynamic_cache import UnifiedDynamicCache as DynamicCache

import asyncio

class MyMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.top_k = 1
        self.queues = [FCFSQueue() for _ in range(self.num_experts)]
    
    async def process_expert_async(self, expert_idx, states):
        # Asynchronously execute expert computation on GPU
        return self.experts[expert_idx](states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Combine reshape and conditional jitter into one operation
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.training and self.jitter_noise > 0:
            hidden_states.mul_(1.0 + (torch.rand_like(hidden_states) * 2 - 1) * self.jitter_noise)

        # Compute routing weights more efficiently
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.div_(routing_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        # Pre-allocate final hidden states tensor
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # Optimize expert routing
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        selected_expert_indices = torch.nonzero(expert_mask.sum(dim=(1, 2))).squeeze(-1)
        
        is_decode = MyCustomMixtral.running_sequences and MyCustomMixtral.running_sequences[0].stage == "decode"

        if is_decode:
            for expert_idx in selected_expert_indices:
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[top_x]
                token_indices = top_x.tolist()
                selected_sequences = [MyCustomMixtral.running_sequences[idx] for idx in token_indices]
                
                # Batch sequence updates
                for seq, state in zip(selected_sequences, current_state):
                    seq.cached_hidden_state = state
                    self.queues[expert_idx].enqueue(seq)
            
            MyCustomMixtral.running_sequences.clear()
            states_list = []
            sequences_list = []

            for expert_idx in range(self.num_experts):
                if self.queues[expert_idx].size() >= 1:
                    expert_sequences = []
                    current_states = []
                    
                    while self.queues[expert_idx].size() > 0:
                        seq = self.queues[expert_idx].dequeue()
                        current_states.append(seq.cached_hidden_state)
                        expert_sequences.append(seq)
                    
                    if current_states:
                        states = torch.stack(current_states)
                        states_list.append((expert_idx, states))
                        sequences_list.extend(expert_sequences)
            
            # Process expert computations asynchronously
            async def compute_all_experts():
                tasks = [self.process_expert_async(expert_idx, states) for expert_idx, states in states_list]
                return await asyncio.gather(*tasks)

            final_states = asyncio.run(compute_all_experts())
            
            MyCustomMixtral.running_sequences.extend(sequences_list)
            final_hidden_states = torch.cat(final_states) if final_states else torch.zeros(
                (0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            batch_size = len(MyCustomMixtral.running_sequences)
            
        else:
            # Process non-decode mode more efficiently
            for expert_idx in selected_expert_indices:
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[top_x]
                current_hidden_states = self.experts[expert_idx](current_state)
                final_hidden_states.index_add_(0, top_x, current_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits


class MyMixtralDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # print(f"Layer index: {self.layer_idx}")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if MyCustomMixtral.running_sequences:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        else:
            present_key_value = None

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        if MyCustomMixtral.running_sequences:
            cached_residuals = residual.clone()

            if use_cache:
                # splited_kv_cache = present_key_value.split_layer_to_caches(self.layer_idx, [seq.kv_cache for seq in MyCustomMixtral.running_sequences])
                splited_kv_cache = present_key_value.split_kv_cache()
            else:
                splited_kv_cache = [None] * len(MyCustomMixtral.running_sequences)

            for seq, cached_residual, kv_cache in zip(MyCustomMixtral.running_sequences, cached_residuals, splited_kv_cache):
                seq.cached_residual = cached_residual
                seq.kv_cache = kv_cache

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)

        if MyCustomMixtral.running_sequences:
            residual = torch.stack([seq.cached_residual for seq in MyCustomMixtral.running_sequences], dim=0)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

class MixtralModel(MixtralModel):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_router_logits = (
                output_router_logits if output_router_logits is not None else self.config.output_router_logits
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

            # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = False
            if use_cache and not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    logger.warning_once(
                        "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                        "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                        "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                    )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            hidden_states = inputs_embeds

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            all_router_logits = () if output_router_logits else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                    
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        cache_position,
                    )
                else:
                    # print(f"Layer: {decoder_layer.layer_idx}, Sequences: {MyCustomMixtral.running_sequences}")
                    if MyCustomMixtral.running_sequences and MyCustomMixtral.running_sequences[0].stage == "decode":
                        # past_key_values = DynamicCache.merge_single_layer([seq.kv_cache for seq in MyCustomMixtral.running_sequences], decoder_layer.layer_idx)
                        past_key_values = DynamicCache([seq.kv_cache for seq in MyCustomMixtral.running_sequences])
                        
                        # # Check if both past_key_values are equal at layer_idx
                        # for idx in range(len(past_key_values.key_cache[decoder_layer.layer_idx])):
                        #     assert torch.allclose(past_key_values.key_cache[decoder_layer.layer_idx][idx], past_key_values_2.key_cache[decoder_layer.layer_idx][0])
                        
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    
                hidden_states = layer_outputs[0]
 
                # if use_cache:
                #     # next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                #     next_decoder_cache = DynamicCache.merge_kv_caches([seq.kv_cache for seq in MyCustomMixtral.running_sequences])

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
 
                if output_router_logits:
                    all_router_logits += (layer_outputs[-1],)

            hidden_states = self.norm(hidden_states)

 
            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            next_cache = DynamicCache([seq.kv_cache for seq in MyCustomMixtral.running_sequences]) if use_cache else None
            
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                    if v is not None
                )
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )

class MyCustomMixtral(MixtralForCausalLM):
    running_sequences: List[Sequence]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralModel(config)
        for i in range(config.num_hidden_layers):
            self.model.layers[i] = MyMixtralDecoderLayer(config, i)
            self.model.layers[i].block_sparse_moe = MyMixtralSparseMoeBlock(config)
        
    def forward(self, batch: Batch, **kwargs):
        MyCustomMixtral.running_sequences = batch.sequences
        input_ids_list, attention_mask_list, past_key_values_list = batch.model_inputs.get_all_inputs()
        
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        if past_key_values_list:
            past_key_values = DynamicCache(past_key_values_list)
        
        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        
        splited_kv_cache = kv_cache.split_kv_cache()
        
        new_batch = Batch(MyCustomMixtral.running_sequences)
        new_batch.update_sequences(logits, splited_kv_cache)
        
        return new_batch
    