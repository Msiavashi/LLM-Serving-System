from abc import ABC, abstractmethod
from typing import List

from src.sequence import Sequence, Stage

class BaseScheduler(ABC):
    @abstractmethod
    def add_sequence_to_queue(self, prompt: str, stage: Stage = Stage.PREFILL) -> None:
        pass
    
    @abstractmethod
    def run_scheduler(self) -> List[Sequence]:
        pass
