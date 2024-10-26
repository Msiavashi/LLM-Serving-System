from src.queues.base_queue import BaseQueue

class ConcurrentQueue(BaseQueue):
    """Thread-safe queue implementation using locks."""
    
    def __init__(self):
        super().__init__()
        self.lock = Lock()

    def enqueue(self, item):
        with self.lock:
            self.queue.append(item)
    
    def dequeue(self):
        with self.lock:
            if self.is_empty():
                raise IndexError("Dequeue from an empty queue.")
            return self.queue.popleft()

    def peek(self):
        with self.lock:
            if self.is_empty():
                raise IndexError("Peek from an empty queue.")
            return self.queue[0]
