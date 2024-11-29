import torch
from typing import Literal

from src.cache.dynamic_cache import DynamicCacheEx
from .sampling import SamplingMetadata


class SequenceBase:
    sequence_id = 0

    def __init__(self, prompt, input_ids, attention_mask, generated_tokens=None, kv_cache=None, device="cuda", sampling_metadata=None):
        self.sequence_id = SequenceBase.sequence_id
        SequenceBase.sequence_id += 1
        self.prompt = prompt
        self.device = device
        self.input_ids = input_ids.squeeze(0).to(self.device)  # Shape: [sequence_length]
        self.attention_mask = attention_mask.squeeze(0).to(self.device)  # Shape: [sequence_length]
        if generated_tokens is not None:
            self.generated_tokens = generated_tokens.to(self.device)
        else:
            self.generated_tokens = torch.empty(0, dtype=self.input_ids.dtype, device=self.device)
        self.kv_cache = kv_cache if kv_cache is not None else DynamicCacheEx()
        self.stage: Literal["prefill", "decode"] = "prefill"
        self.sampling_metadata = sampling_metadata if sampling_metadata is not None else SamplingMetadata(num_tokens=10)

    def update(self, next_token_ids, new_kv_cache):
        next_token_ids = next_token_ids.to(self.device)
        self.generated_tokens = torch.cat([self.generated_tokens, next_token_ids], dim=-1)
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones_like(next_token_ids, device=self.device)], dim=-1
        )
        self.kv_cache = new_kv_cache

    def get_generated_text(self, tokenizer):
        return tokenizer.decode(self.generated_tokens, skip_special_tokens=True)

    def __str__(self):
        return (
            f"Sequence(sequence_id={self.sequence_id}, prompt={self.prompt}, generated_text={self.generated_tokens})"
        )

    def __repr__(self):
        return str(self)

class GenerationConfig:
    def __init__(self, max_tokens=10):
        self.max_tokens = max_tokens

class Sequence(SequenceBase):
    def __init__(self, prompt, tokenizer, generation_config: GenerationConfig = None, device="cuda"):
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.cached_hidden_state = None
        self.routing_weights_cache = {}
        self.expert_outputs_cache = {}
        self.cached_residual = None
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        super().__init__(prompt, input_ids, attention_mask, kv_cache=None, device=device)