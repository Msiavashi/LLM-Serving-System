from src.queues.base_queue import BaseQueue

class FCFSQueue(BaseQueue):
    """First-Come, First-Served (FCFS) queue implementation."""
    
    def _storage_enqueue(self, packed_item):
        self.queue.append(packed_item)
    
    def _storage_dequeue(self):
        return self.queue.pop(0)

    def _storage_peek(self):
        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)