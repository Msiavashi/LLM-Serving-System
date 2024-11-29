from typing import Tuple

import torch
import torch.nn.functional as F


class SequenceProcessor:
    @staticmethod
    def pad_sequence(sequence, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_length = max_length - sequence.input_ids.size(0)
        if padding_length > 0:
            padded_input_ids = F.pad(
                sequence.input_ids, 
                (0, padding_length), 
                value=sequence.tokenizer.pad_token_id
            )
            padded_attention_mask = F.pad(
                sequence.attention_mask, 
                (0, padding_length), 
                value=0
            )
            return padded_input_ids, padded_attention_mask
        return sequence.input_ids, sequence.attention_mask

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