from src.queues.base_queue import BaseQueue
import time

class FCFSQueue(BaseQueue):
    """First-Come, First-Served (FCFS) queue implementation."""
    
    def enqueue(self, item):
        self.queue.append((item, time.time()))
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue.")
        return self.queue.pop(0)[0]

    def peek(self):
        """Return the item at the front of the queue without removing it as tuple (object, insertion_timestamp)."""
        if self.is_empty():
            raise IndexError("Peek from an empty queue.")
        return self.queue[0]