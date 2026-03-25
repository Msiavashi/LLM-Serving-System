"""Base Sequence Module for language model inference."""
import time
import itertools
import torch
from typing import Optional, Any

from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
from src.samplers.sampling_metadata import SamplingMetadata
from .stage import Stage


_sequence_id_counter = itertools.count()


class SequenceBase:

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
        self.sequence_id = next(_sequence_id_counter)

        self.prompt = prompt
        self.device = device
        self.priority = priority

        self.input_ids = input_ids.squeeze(0).to(self.device)
        self.attention_mask = attention_mask.squeeze(0).to(self.device)

        self.sampling_metadata = sampling_metadata if sampling_metadata is not None else SamplingMetadata(num_tokens=20)
        max_new = self.sampling_metadata.max_sequence_length

        if generated_tokens is not None:
            self.generated_tokens = generated_tokens.to(self.device)
        else:
            self.generated_tokens = torch.empty(0, dtype=self.input_ids.dtype, device=self.device)

        # Pre-allocate buffers to avoid O(n²) torch.cat in update()
        self._gen_buffer = torch.zeros(max_new, dtype=self.input_ids.dtype, device=self.device)
        self._gen_count = 0
        self._attn_buffer = torch.ones(
            self.input_ids.shape[0] + max_new, dtype=self.attention_mask.dtype, device=self.device
        )
        self._attn_buffer[:self.input_ids.shape[0]] = self.attention_mask
        self._attn_count = self.input_ids.shape[0]

        self.kv_cache = kv_cache if kv_cache is not None else DynamicCache()
        self.stage: Stage = Stage.PREFILL

        # Timing fields
        self.arrival_time = time.time()
        self.first_token_time = None
        self.finish_time = None
        self.previous_token_time = time.time()

    def update(self, next_token_ids: torch.Tensor, new_kv_cache: Any) -> None:
        next_token_ids = next_token_ids.to(self.device)
        if next_token_ids.numel() > 0:
            # Track first token timing
            if self._gen_count == 0:
                self.first_token_time = time.time()

            # Write to pre-allocated buffer (O(1) instead of O(n) concat)
            n = next_token_ids.numel()
            end = self._gen_count + n
            if end <= self._gen_buffer.shape[0]:
                self._gen_buffer[self._gen_count:end] = next_token_ids
            else:
                # Fallback: extend buffer if needed
                self._gen_buffer = torch.cat([
                    self._gen_buffer,
                    torch.zeros(max(n, self._gen_buffer.shape[0]), dtype=self._gen_buffer.dtype, device=self.device)
                ])
                self._gen_buffer[self._gen_count:end] = next_token_ids
            self._gen_count = end

            # Update attention mask buffer
            attn_end = self._attn_count + n
            if attn_end <= self._attn_buffer.shape[0]:
                self._attn_buffer[self._attn_count:attn_end] = 1
            else:
                self._attn_buffer = torch.cat([
                    self._attn_buffer,
                    torch.ones(max(n, self._attn_buffer.shape[0]), dtype=self._attn_buffer.dtype, device=self.device)
                ])
            self._attn_count = attn_end

            # Keep generated_tokens in sync (view into buffer)
            self.generated_tokens = self._gen_buffer[:self._gen_count]
            self.attention_mask = self._attn_buffer[:self._attn_count]

            self.sampling_metadata.increment_token_count()

        self.kv_cache = new_kv_cache

    def get_generated_text(self, tokenizer) -> str:
        return tokenizer.decode(self.generated_tokens, skip_special_tokens=True)

    def get_input_prompt_length(self) -> int:
        return self.input_ids.size(0)

    def get_total_sequence_length(self) -> int:
        return self.get_input_prompt_length() + self._gen_count

    def is_finished(self) -> bool:
        return self.sampling_metadata.is_finished()

    def __str__(self) -> str:
        return (
            f"Sequence(sequence_id={self.sequence_id}, priority={self.priority}, "
            f"prompt={self.prompt}, generated_text={self.generated_tokens})"
        )

    def __repr__(self) -> str:
        return str(self)
