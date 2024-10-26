from abc import ABC, abstractmethod
from collections import deque
from threading import Lock

class BaseQueue(ABC):
    """Abstract base class for queue implementations."""
    
    def __init__(self):
        self.queue = deque()
    
    @abstractmethod
    def enqueue(self, item):
        """Insert an item into the queue."""
        pass
    
    @abstractmethod
    def dequeue(self):
        """Remove and return an item from the queue."""
        pass

    @abstractmethod
    def peek(self):
        """Return the item at the front of the queue without removing it."""
        pass

    def is_empty(self):
        """Return True if the queue is empty, else False."""
        return len(self.queue) == 0

    def size(self):
        """Return the number of items in the queue."""
        return len(self.queue)

