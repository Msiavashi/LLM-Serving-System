from typing import Tuple

import torch
import torch.nn.functional as F


class SequenceProcessor:
    @staticmethod
    def pad_sequence(sequence, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = 16
        input_ids = sequence.input_ids[:max_length]
        attention_mask = sequence.attention_mask[:max_length]
        padding_length = max_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = F.pad(
                input_ids, 
                (0, padding_length), 
                value=sequence.tokenizer.pad_token_id
            )
            attention_mask = F.pad(
                attention_mask, 
                (0, padding_length), 
                value=0
            )
        return input_ids, attention_mask

    @staticmethod
    def process_prefill(sequence, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, any]:
        input_ids, attention_mask = SequenceProcessor.pad_sequence(sequence, max_length)
        return (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            sequence.kv_cache
        )

    @staticmethod
    def process_decode(sequence) -> Tuple[torch.Tensor, torch.Tensor, any]:
        return (
            sequence.generated_tokens[-1:].unsqueeze(0),
            sequence.attention_mask[-1:].unsqueeze(0),
            sequence.kv_cache
        )