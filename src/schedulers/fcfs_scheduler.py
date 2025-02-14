import time
from src.schedulers.base_fcfs_scheduler import BaseFCFSScheduler

class FCFSScheduler(BaseFCFSScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        super().__init__(engine, tokenizer, batch_size)

    def run_scheduler(self):
        return self.run_loop(lambda batch: self.engine.run_batch(batch))