from typing import List
import torch
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralDecoderLayer, MixtralConfig, MixtralRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from torch.nn import functional as F
from src.queues import FCFSQueue
from src.sequence import Sequence
from src.batching.batch import Batch
import time
from typing import Optional, Tuple
from torch import nn
from transformers.cache_utils import Cache
from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
import math
import copy


class MyMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    counter = 0
    def __init__(self, config):
        super().__init__(config)
        self.top_k = 1
        self.queues = [FCFSQueue() for _ in range(self.num_experts)]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        MyMixtralSparseMoeBlock.counter += 1
        # print(MyMixtralSparseMoeBlock.counter % 32)
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
             
            if MyCustomMixtral.running_sequences and MyCustomMixtral.running_sequences[0].stage == "decode" and False:
                new_top_x = torch.empty(0, dtype=torch.long, device=current_state.device)
                # 0. Get the indices of the tokens that goes to this expert_idx
                token_indices = [int(idx) for idx in top_x.tolist()]
                
                selected_sequences = [MyCustomMixtral.running_sequences[idx] for idx in token_indices]
                
                # 1. Cache the hidden_state and routing_weights for the sequence
                remove_indices = []

                for i, token_idx in enumerate(token_indices):
                    seq = MyCustomMixtral.running_sequences[token_idx]
                    seq.cached_hidden_state = current_state[i]
                    seq.cached_routing_weight = routing_weights[token_idx]
                    
                    # Collect indices for removal after the loop
                    remove_indices.append(i)
 
                # Use boolean masks to exclude indices rather than torch.cat
                mask_current_state = torch.tensor([i not in remove_indices for i in range(current_state.size(0))], device=current_state.device)
                mask_routing_weights = torch.tensor([i not in token_indices for i in range(routing_weights.size(0))], device=routing_weights.device)

                # Apply the mask to filter out unwanted elements
                current_state = current_state[mask_current_state]
                routing_weights = routing_weights[mask_routing_weights]

                # 2. Add the sequence to the expert's queue
                for seq in selected_sequences:
                    self.queues[expert_idx].enqueue(seq)
                
                # 3. Remove the seuqence from the running_sequences list 
                MyCustomMixtral.running_sequences = [seq for seq in MyCustomMixtral.running_sequences if seq not in selected_sequences]
                
                # 4. If the queue is full, pop 4 sequences from the queue and update the sequence's hidden_state and routing_weights 
                if self.queues[expert_idx].size() >= 1:
                    for _ in range(self.queues[expert_idx].size()):
                        seq = self.queues[expert_idx].dequeue()
                        
                        # add seq.cached_hidden_state to the hidden_state
                        current_state = torch.cat([current_state, seq.cached_hidden_state[None]], dim=0)
                        # add seq.cached_routing_weight to the routing_weights
                        routing_weights = torch.cat([routing_weights, seq.cached_routing_weight[None]], dim=0)
                        
                        # Add the sequence to the running_sequences list
                        MyCustomMixtral.running_sequences.append(seq)
                        
                        # 5. remove the cached hidden_state and routing_weights from the sequence
                        seq.cached_hidden_state = None
                        seq.cached_routing_weight = None
                        # Append the index of the newly added routing weight to new_top_x
                        new_top_x = torch.cat([new_top_x, torch.tensor([routing_weights.size(0) - 1], dtype=torch.long, device=current_state.device)], dim=0)
                top_x = new_top_x

            # current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            current_hidden_states = expert_layer(current_state)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MyMixtralDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config ,layer_idx)

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
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states.clone()
        
        # [Added] Caching the residual for the sequence
        # for i, seq in enumerate(MyCustomMixtral.running_sequences):
        #     seq.cashed_residual = residual[i]
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        
        # [Added] Restore the residual for the sequence
        # residual = torch.empty_like(hidden_states)
        # for i, seq in enumerate(MyCustomMixtral.running_sequences):
        #     residual[i] = seq.cashed_residual
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

class MixtralAttention(nn.Module):
    def __init__(self, config: MixtralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MyCustomMixtral(MixtralForCausalLM):
    running_sequences: List[Sequence]
    
    def __init__(self, config):
        super().__init__(config)
        for i in range(config.num_hidden_layers):
            self.model.layers[i] = MyMixtralDecoderLayer(config, i)
            self.model.layers[i].block_sparse_moe = MyMixtralSparseMoeBlock(config)
            self.model.layers[i].self_attn = MixtralAttention(config, i)
        
    def forward(self, batch: Batch, **kwargs):
        MyCustomMixtral.running_sequences = batch.sequences
        input_ids_list, attention_mask_list, past_key_values_list = batch.model_inputs.get_all_inputs()
        
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        if past_key_values_list:
            past_key_values = DynamicCache.merge_kv_caches(past_key_values_list)
        else:
            past_key_values = DynamicCache()

        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        
        splited_kv_cache = kv_cache.split_kv_cache(len(MyCustomMixtral.running_sequences))
        
        new_batch = Batch(MyCustomMixtral.running_sequences)
        new_batch.update_sequences(logits, splited_kv_cache)
        
        return new_batch
