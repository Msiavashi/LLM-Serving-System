"""Batch Processing Module for efficient sequence handling in inference."""
from typing import List, Union, Tuple, Optional, Any
from dataclasses import dataclass
import torch
from .utils import SequenceProcessor
from src.sequence import Sequence, Stage


@dataclass
class ModelInputs:
    input_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    past_key_values: List[Any]

    def get_all_inputs(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Any]]:
        return self.input_ids, self.attention_masks, self.past_key_values

    def is_empty(self) -> bool:
        return len(self.input_ids) == 0

    def clear(self) -> None:
        self.input_ids = []
        self.attention_masks = []
        self.past_key_values = []

class Batch:
    def __init__(self, sequences: Optional[List["Sequence"]] = None):
        self._sequences = sequences or []
        self._model_inputs = ModelInputs([], [], [])
        self._processor = SequenceProcessor()

    @property
    def sequences(self) -> List["Sequence"]:
        return self._sequences

    @sequences.setter
    def sequences(self, sequences: List["Sequence"]) -> None:
        self._sequences = sequences

    def clear(self) -> None:
        self._sequences = []
        self._model_inputs.clear()

    def is_decode(self) -> bool:
        return self.sequences and self.sequences[0].stage == Stage.DECODE

    def is_prefill(self) -> bool:
        return self.sequences and self.sequences[0].stage == Stage.PREFILL

    def get_stage(self) -> Optional[str]:
        return self.sequences[0].stage if self.sequences else None

    def add_sequence(self, sequence: Union["Sequence", List["Sequence"]]) -> None:
        if isinstance(sequence, list):
            self._sequences.extend(sequence)
        else:
            self._sequences.append(sequence)

    def size(self) -> int:
        return len(self._sequences)

    def is_empty(self) -> bool:
        return not self._sequences

    def _preprocess_sequences(self) -> None:
        if not self._sequences:
            return

        input_ids_list, attention_mask_list, past_key_values_list = [], [], []

        for sequence in self._sequences:
            if sequence.stage == Stage.PREFILL:
                inputs = self._processor.process_prefill(sequence)
            else:
                inputs = self._processor.process_decode(sequence)

            input_ids, attention_mask, past_key_values = inputs
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            past_key_values_list.append(past_key_values)

        self._model_inputs = ModelInputs(input_ids_list, attention_mask_list, past_key_values_list)

    @property
    def model_inputs(self) -> ModelInputs:
        self._preprocess_sequences()
        return self._model_inputs

    def update_sequences(self, logits: torch.Tensor, kv_caches: List[Any]) -> None:
        if len(self._sequences) != logits.shape[0]:
            raise ValueError(f"Number of sequences ({len(self._sequences)}) does not match logits batch size ({logits.shape[0]})")

        for i, sequence in enumerate(self._sequences):
            last_token_logits = logits[i, -1, :]
            next_token_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            sequence.update(next_token_ids, kv_caches[i])

            if sequence.stage == Stage.PREFILL:
                sequence.stage = Stage.DECODE

    @staticmethod
    def update_kv_caches(sequences: List["Sequence"], kv_caches: List[Any]) -> None:
        if len(sequences) != len(kv_caches):
            raise ValueError(f"Number of sequences ({len(sequences)}) does not match number of KV caches ({len(kv_caches)})")

        for sequence, kv_cache in zip(sequences, kv_caches):
            sequence.kv_cache = kv_cache

    @staticmethod
    def get_kv_caches(sequences: List["Sequence"]) -> List[Any]:
        return [sequence.kv_cache for sequence in sequences]

    def __await__(self):
        yield self
        return self