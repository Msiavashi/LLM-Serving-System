from typing import List, Union, Tuple
from dataclasses import dataclass
import torch
from .utils import SequenceProcessor
from src.sequence import Sequence, Stage

@dataclass
class ModelInputs:
    input_ids: List
    attention_masks: List
    past_key_values: List

    def get_all_inputs(self) -> Tuple:
        return self.input_ids, self.attention_masks, self.past_key_values

class Batch:
    def __init__(self, sequences: List["Sequence"] = None):
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
        self._model_inputs = ModelInputs([], [], [])
    
    def is_decode(self) -> bool:
        return self.sequences and self.sequences[0].stage == Stage.DECODE
    
    def is_prefill(self) -> bool:
        return self.sequences and self.sequences[0].stage == Stage.PREFILL
    
    def get_stage(self) -> Stage:
        return self.sequences[0].stage if self.sequences else None

    def add_sequence(self, sequence: Union["Sequence", List["Sequence"]]) -> None:
        if isinstance(sequence, list):
            self.sequences.extend(sequence)
        else:
            self.sequences.append(sequence)

    def size(self) -> int:
        return len(self.sequences)
    
    def is_empty(self) -> bool:
        return not self.sequences

    def _preprocess_sequences(self) -> None:
        if not self.sequences:
            return

        input_ids_list, attention_mask_list, past_key_values_list = [], [], []
        max_length = max(seq.input_ids.size(0) for seq in self.sequences)

        for sequence in self.sequences:
            if sequence.stage == Stage.PREFILL:
                inputs = self._processor.process_prefill(sequence, max_length)
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

    def update_sequences(self, logits: torch.Tensor, kv_caches: List) -> None:
        for i, sequence in enumerate(self.sequences):
            last_token_logits = logits[i, -1, :]
            next_token_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            
            sequence.update(next_token_ids, kv_caches[i])
            
            if sequence.stage == Stage.PREFILL:
                sequence.stage = Stage.DECODE

    @staticmethod
    def update_kv_caches(sequences: List["Sequence"], kv_caches: List) -> None:
        for sequence, kv_cache in zip(sequences, kv_caches):
            sequence.kv_cache = kv_cache

    @staticmethod
    def get_kv_caches(sequences: List["Sequence"]) -> List:
        return [sequence.kv_cache for sequence in sequences]

    def __await__(self):
        # Make the batch awaitable by returning itself
        yield self
        return self