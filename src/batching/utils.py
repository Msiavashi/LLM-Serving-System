from typing import Tuple

import torch
import torch.nn.functional as F


class SequenceProcessor:

    @staticmethod
    def process_prefill(sequence) -> Tuple[torch.Tensor, torch.Tensor, any]:
        return (
            sequence.input_ids.unsqueeze(0),
            sequence.attention_mask.unsqueeze(0),
            sequence.kv_cache
        )

    @staticmethod
    def process_decode(sequence) -> Tuple[torch.Tensor, torch.Tensor, any]:
        return (
            sequence.generated_tokens[-1:].unsqueeze(0),
            sequence.attention_mask[-1:].unsqueeze(0),
            sequence.kv_cache
        )