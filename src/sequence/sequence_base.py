"""Base Sequence Module for language model inference."""
import torch
from typing import Optional, Any
import numpy as np

from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
from src.samplers.sampling_metadata import SamplingMetadata
from .stage import Stage


class SequenceBase:
    sequence_id_counter = 0

    def __init__(
        self, 
        prompt: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_tokens: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        device: str = "cuda",
        sampling_metadata: Optional[SamplingMetadata] = None,
        priority: int = 0
    ):
        self.sequence_id = SequenceBase.sequence_id_counter
        SequenceBase.sequence_id_counter += 1
        
        self.prompt = prompt
        self.device = device
        self.priority = priority
        
        self.input_ids = input_ids.squeeze(0).to(self.device)
        self.attention_mask = attention_mask.squeeze(0).to(self.device)
        
        if generated_tokens is not None:
            self.generated_tokens = generated_tokens.to(self.device)
        else:
            self.generated_tokens = torch.empty(0, dtype=self.input_ids.dtype, device=self.device)
        
        self.kv_cache = kv_cache if kv_cache is not None else DynamicCache()
        self.stage: Stage = Stage.PREFILL
        np.random.seed(42)  
        self.sampling_metadata = sampling_metadata if sampling_metadata is not None else SamplingMetadata(num_tokens=20)

    def update(self, next_token_ids: torch.Tensor, new_kv_cache: Any) -> None:
        next_token_ids = next_token_ids.to(self.device)
        self.generated_tokens = torch.cat([self.generated_tokens, next_token_ids], dim=-1)
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones_like(next_token_ids, device=self.device)], 
            dim=-1
        )
        self.kv_cache = new_kv_cache
 
    def get_generated_text(self, tokenizer) -> str:
        return tokenizer.decode(self.generated_tokens, skip_special_tokens=True)

    def get_input_prompt_length(self) -> int:
        return self.input_ids.size(0)

    def get_total_sequence_length(self) -> int:
        return self.get_input_prompt_length() + self.generated_tokens.size(0)
        
    def is_finished(self) -> bool:
        current_count = self.sampling_metadata.current_token_count if hasattr(self.sampling_metadata, 'current_token_count') else 0
        max_length = self.sampling_metadata.max_sequence_length if hasattr(self.sampling_metadata, 'max_sequence_length') else float('inf')
        
        return current_count >= max_length

    def __str__(self) -> str:
        return (
            f"Sequence(sequence_id={self.sequence_id}, priority={self.priority}, "
            f"prompt={self.prompt}, generated_text={self.generated_tokens})"
        )

    def __repr__(self) -> str:
        return str(self)