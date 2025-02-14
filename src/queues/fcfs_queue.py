from src.queues.storage.base_queue_storage import BaseQueueStorage
from src.queues.storage.memory_storage import MemoryQueueStorage
import time

class FCFSQueue:
    """First-Come, First-Served (FCFS) queue implementation."""
    
    def __init__(self, storage: BaseQueueStorage = None):
        self.storage = storage or MemoryQueueStorage()

    def _pack_item(self, item):
        return (item, time.time())

    def _unpack_item(self, packed_item):
        return packed_item[0] if packed_item else None

    def enqueue(self, item):
        packed_item = self._pack_item(item)
        self.storage.enqueue(packed_item)
        
    def enqueue_many(self, items):
        packed_items = [self._pack_item(item) for item in items]
        self.storage.enqueue_many(packed_items)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue.")
        packed_item = self.storage.dequeue()
        return self._unpack_item(packed_item)
    
    def dequeue_all(self):
        packed_items = self.storage.dequeue_all()
        return [self._unpack_item(item) for item in packed_items]

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty queue.")
        return self.storage.peek()

    def is_empty(self):
        return self.storage.is_empty()

    def size(self):
        return self.storage.size()