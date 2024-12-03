# TODO: This model is implemented to measure the latency of running attention on CPU vs GPU.
# The MyLlamaAttention class could be removed to match the default implementation.


from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, apply_rotary_pos_emb, repeat_kv
from src.batching.batch import Batch
from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
import time
import math

class MyLlamaAttention(LlamaAttention):
    
    def original_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Move everything to the CPU
        hidden_states = hidden_states.cpu()
        attention_mask = attention_mask.cpu()
        position_ids = position_ids.cpu()
        cache_position = cache_position.cpu()
        self.cpu()
        past_key_value.transfer_layer_to(self.layer_idx, torch.device("cpu"))
        if position_embeddings is not None:
            position_embeddings = tuple(pe.cpu() for pe in position_embeddings)
        
        start_time = time.time()
        hidden_states, attention_weights, past_key_value = self.original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Latency: {latency:.2f} ms")
        
        # Move resutls back to the GPU
        hidden_states = hidden_states.cuda() if hidden_states is not None else None
        attention_weights = attention_weights.cuda() if attention_weights is not None else None
        # past_key_value = past_key_value.transfer_layer_to(self.layer_idx, torch.device("cuda"))
        return hidden_states, attention_weights, past_key_value


class Llama8B(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # override the attention mechanism
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = MyLlamaAttention(config, layer_idx=i)
    
    
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
