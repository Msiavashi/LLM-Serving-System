"""Batch: Container for grouping sequences during scheduling."""
from typing import List, Union, Optional
from src.sequence import Sequence, Stage


class Batch:
    """Groups sequences for batch processing by the engine.

    In v2, the Batch is a simple container — input preparation and sampling
    are handled by QllmEngine and SamplingProcessor respectively.
    """

    def __init__(self, sequences: Optional[List["Sequence"]] = None):
        self._sequences = sequences or []

    @property
    def sequences(self) -> List["Sequence"]:
        return self._sequences

    @sequences.setter
    def sequences(self, sequences: List["Sequence"]) -> None:
        self._sequences = sequences

    def clear(self) -> None:
        self._sequences = []

    def is_decode(self) -> bool:
        return bool(self.sequences) and self.sequences[0].stage == Stage.DECODE

    def is_prefill(self) -> bool:
        return bool(self.sequences) and self.sequences[0].stage == Stage.PREFILL

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

    def __await__(self):
        yield self
        return self
