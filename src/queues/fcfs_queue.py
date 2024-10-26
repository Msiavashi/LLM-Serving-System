from src.queues.base_queue import BaseQueue

class FCFSQueue(BaseQueue):
    """First-Come, First-Served (FCFS) queue implementation."""
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue.")
        return self.queue.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty queue.")
        return self.queue[0]