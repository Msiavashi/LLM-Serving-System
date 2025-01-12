from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

class BaseQueueStorage(ABC):
    @abstractmethod
    def enqueue(self, packed_item: Tuple[Any, float]) -> None:
        """Store an item in the queue."""
        pass

    @abstractmethod
    def dequeue(self) -> Optional[Tuple[Any, float]]:
        """Remove and return an item from the queue."""
        pass

    @abstractmethod
    def peek(self) -> Optional[Tuple[Any, float]]:
        """Return the next item without removing it."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of items in the queue."""
        pass