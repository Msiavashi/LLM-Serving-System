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

    def update_sequences(self, logits: torch.Tensor, kv_caches: List[Any], temperature: float = 0.0) -> None:
        if len(self._sequences) != logits.shape[0]:
            raise ValueError(f"Number of sequences ({len(self._sequences)}) does not match logits batch size ({logits.shape[0]})")

        top_k = 50          # widen candidate pool
        top_p = 0.9         # nucleus threshold
        freq_penalty = 0.2  # frequency penalty coefficient

        exclude_token_ids = getattr(self, "exclude_token_ids", getattr(type(self), "exclude_token_ids", [0]))
        leading_exclude_token_ids = getattr(self, "leading_exclude_token_ids",
                                            getattr(type(self), "leading_exclude_token_ids", []))

        exclude_tensor = torch.tensor(exclude_token_ids, device=logits.device)

        for i, sequence in enumerate(self._sequences):
            last_token_logits = logits[i, -1, :].clone()

            # Apply simple frequency penalty to reduce repeated punctuation bursts
            if sequence.generated_tokens.numel() > 0:
                unique_tokens, counts = torch.unique(sequence.generated_tokens, return_counts=True)
                penalty = torch.zeros_like(last_token_logits)
                penalty.index_add_(0, unique_tokens.to(last_token_logits.device),
                                   counts.to(last_token_logits.dtype) * freq_penalty)
                last_token_logits = last_token_logits - penalty  # subtract to lower repeated tokens

            # Remove globally excluded tokens
            if exclude_tensor.numel() > 0:
                last_token_logits[exclude_tensor] = -float("inf")

            # Top-k
            values, indices = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
            # Convert to probs
            probs = torch.softmax(values, dim=-1)

            # Top-p (nucleus) filtering on the top-k slice
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = cumulative <= top_p
            # Always keep at least one token
            if not nucleus_mask.any():
                nucleus_mask[0] = True
            kept_sorted_idx = sorted_idx[nucleus_mask]
            kept_indices = indices[kept_sorted_idx]
            kept_values = last_token_logits[kept_indices]
            kept_probs = torch.softmax(kept_values, dim=-1)

            # Filter leading unwanted punctuation tokens only for very first generated token(s)
            if sequence.generated_tokens.numel() == 0 and leading_exclude_token_ids:
                leading_ex_tensor = torch.tensor(leading_exclude_token_ids, device=kept_indices.device)
                leading_mask = ~torch.isin(kept_indices, leading_ex_tensor)
                if leading_mask.any():
                    kept_indices = kept_indices[leading_mask]
                    kept_probs = kept_probs[leading_mask]
                # if all filtered, fall back (keep original kept_indices / kept_probs)

            # Sample
            if kept_indices.numel() == 0:
                # Absolute fallback to argmax over original logits (after penalties)
                next_token_ids = torch.argmax(last_token_logits).unsqueeze(0)
            else:
                sampled_local = torch.multinomial(kept_probs, num_samples=1)
                next_token_ids = kept_indices[sampled_local].unsqueeze(0).flatten()

            sequence.update(next_token_ids, kv_caches[i])

            if sequence.stage == Stage.PREFILL:
                sequence.stage = Stage.DECODE

    @staticmethod
    def update_kv_caches(sequences: List["Sequence"], kv_caches: List[Any]) -> None:
        if len(sequences) != len(kv_caches):
            raise ValueError(f"Number of sequences ({len(sequences)}) does not match number of KV caches ({len(kv_caches)})")

        for sequence, kv_cache in zip(sequences, kv_caches):
            # Fix: Use sequence.update to ensure proper cache and token management
            sequence.update(torch.empty(0, dtype=sequence.input_ids.dtype, device=sequence.device), kv_cache)

    @staticmethod
    def get_kv_caches(sequences: List["Sequence"]) -> List[Any]:
        return [sequence.kv_cache for sequence in sequences]

    def __await__(self):
        yield self
        return self