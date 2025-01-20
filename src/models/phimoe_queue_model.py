from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PhimoeForCausalLM, PhimoeConfig
from transformers.models.phimoe.modeling_phimoe import (
    PhimoeSparseMoeBlock, PhimoeDecoderLayer, PhimoeModel, 
    MoeModelOutputWithPast, PhimoeAttention
)
from src.mixins.model_input_mixin import ModelInputMixin
from src.mixins.model_output_mixin import ModelOutputMixin
from src.mixins.sparse_moe_block_with_queue_mixin import SparseMoeBlockWithQueuesMixin
from src.batching.batch import Batch
from src.cache.unified_dynamic_cache import UnifiedDynamicCache as DynamicCache
from transformers.cache_utils import Cache


class MyPhimoeSparseMoeBlock(PhimoeSparseMoeBlock, SparseMoeBlockWithQueuesMixin):
    def __init__(self, config):
        PhimoeSparseMoeBlock.__init__(self, config)
        SparseMoeBlockWithQueuesMixin.__init__(self, self.num_experts)

    def forward(self, hidden_states: torch.Tensor, running_batch) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.training and self.input_jitter_noise > 0:
            hidden_states.mul_(1.0 + (torch.rand_like(hidden_states) * 2 - 1) * self.input_jitter_noise)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.div_(routing_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        selected_expert_indices = torch.nonzero(expert_mask.sum(dim=(1, 2))).squeeze(-1)
        
        if running_batch.is_decode():
            final_hidden_states = self.process_decode(hidden_states, expert_mask, selected_expert_indices, running_batch, hidden_dim)
            batch_size = running_batch.size()
        else:
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), 
                dtype=hidden_states.dtype, 
                device=hidden_states.device
            )
            final_hidden_states = self.process_prefill(hidden_states, expert_mask, selected_expert_indices, final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits


class MyPhimoeDecoderLayer(PhimoeDecoderLayer):
    def __init__(self, config: PhimoeConfig, layer_idx: int):
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
        running_batch: Batch = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if not running_batch.is_empty():
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        else:
            present_key_value = None

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not running_batch.is_empty():
            cached_residuals = residual

            if use_cache:
                splited_kv_cache = present_key_value.split_kv_cache()
            else:
                splited_kv_cache = [None] * running_batch.size()

            for seq, cached_residual, kv_cache in zip(running_batch.sequences, cached_residuals, splited_kv_cache):
                seq.cached_residual = cached_residual
                seq.kv_cache = kv_cache

        hidden_states, router_logits = self.block_sparse_moe(hidden_states, running_batch)

        if not running_batch.is_empty():
            residual = torch.stack([seq.cached_residual for seq in running_batch.sequences], dim=0)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class PhiMoeModel(PhimoeModel):
    def __init__(self, config):
        super().__init__(config)
        self.running_batch: Batch = None
        
    def set_running_batch(self, batch):
        self.running_batch = batch

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
    ) -> Union[Tuple, MoeModelOutputWithPast]:
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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

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

        position_embeddings = self.rotary_emb(hidden_states, seq_len=cache_position[-1] + 1)

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
                    position_embeddings,
                )
            else:
                if self.running_batch and self.running_batch.is_decode():
                    past_key_values = DynamicCache([seq.kv_cache for seq in self.running_batch.sequences])
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    running_batch=self.running_batch,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = DynamicCache([seq.kv_cache for seq in self.running_batch.sequences]) if use_cache else None
        
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


class PhiMoe(PhimoeForCausalLM, ModelInputMixin, ModelOutputMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = PhiMoeModel(config)
        self._initialize_layers(config)
        ModelInputMixin.__init__(self)
        
    def forward(self, batch: Batch, **kwargs) -> Batch:
        input_ids, attention_mask, past_key_values, running_batch = self._prepare_inputs(batch)

        self.model.set_running_batch(running_batch)
        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)

        self._update_batch(outputs, running_batch)

        return self.running_batch
    
    def _initialize_layers(self, config):
        for i in range(config.num_hidden_layers):
            self.model.layers[i] = MyPhimoeDecoderLayer(config, i)
            self.model.layers[i].block_sparse_moe = MyPhimoeSparseMoeBlock(config)
