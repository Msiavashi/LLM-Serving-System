from abc import ABC, abstractmethod
from typing import List
from src.sequence import Sequence

class BaseEngine(ABC):
    @abstractmethod
    def run_batch(self, batch) -> List[Sequence]:
        pass
