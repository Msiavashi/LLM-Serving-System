from typing import List, Optional, Any, Tuple
from src.queues.storage.base_queue_storage import BaseQueueStorage

class MemoryQueueStorage(BaseQueueStorage):
    def __init__(self):
        self.items = []

    def enqueue(self, packed_item: Tuple[Any, float]) -> None:
        self.items.append(packed_item)
    
    def enqueue_many(self, packed_items: List[Tuple[Any, float]]) -> None:
        self.items.extend(packed_items)

    def dequeue(self) -> Optional[Tuple[Any, float]]:
        return self.items.pop(0) if not self.is_empty() else None
    
    def dequeue_all(self) -> List[Tuple[Any, float]]:
        items = self.items
        self.items = []
        return items

    def peek(self) -> Optional[Tuple[Any, float]]:
        return self.items[0] if not self.is_empty() else None

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def size(self) -> int:
        return len(self.items)
