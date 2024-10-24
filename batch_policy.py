class BasePolicy:
    def __init__(self, batch_size, scheduler):
        self.batch_size = batch_size
        self.scheduler = scheduler

class FCFSPolicy(BasePolicy):
    def __init__(self, batch_size, scheduler):
        super().__init__(batch_size, scheduler)
        
    def get_next_batch(self):
        batch = []
        while not self.scheduler.queue.is_empty() and len(batch) < self.batch_size:
            batch.append(self.scheduler.queue.dequeue())
        return batch
            
    