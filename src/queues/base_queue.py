from abc import ABC, abstractmethod
import time

class BaseQueue(ABC):
    """Abstract base class for queue implementations."""
    
    def __init__(self):
        self.queue = []

    def _pack_item(self, item):
        """Pack an item with its timestamp"""
        return (item, time.time())
    
    def _unpack_item(self, packed_item):
        """Unpack an item tuple into item and timestamp"""
        return packed_item[0] if packed_item else None

    @abstractmethod
    def _storage_enqueue(self, packed_item):
        """Storage-specific enqueue implementation"""
        pass
    
    @abstractmethod
    def _storage_dequeue(self):
        """Storage-specific dequeue implementation"""
        pass

    @abstractmethod
    def _storage_peek(self):
        """Storage-specific peek implementation"""
        pass
    
    def enqueue(self, item):
        """Insert an item into the queue."""
        packed_item = self._pack_item(item)
        self._storage_enqueue(packed_item)
    
    def dequeue(self):
        """Remove and return an item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue.")
        packed_item = self._storage_dequeue()
        return self._unpack_item(packed_item)

    def peek(self):
        """Return the item at the front of the queue without removing it."""
        if self.is_empty():
            raise IndexError("Peek from an empty queue.")
        return self._storage_peek()

    @abstractmethod
    def is_empty(self):
        """Return True if the queue is empty, else False."""
        pass

    @abstractmethod
    def size(self):
        """Return the number of items in the queue."""
        pass
