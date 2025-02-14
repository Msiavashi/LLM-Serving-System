import time
from src.schedulers.base_fcfs_scheduler import BaseFCFSScheduler

class FCFSScheduler(BaseFCFSScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        super().__init__(engine, tokenizer, batch_size)

    def run_scheduler(self):
        finished_sequences = []
        
        while not (self.decode_queue.is_empty() and self.prefill_queue.is_empty()):
             
            batch, is_decode = self._get_next_batch()
             
            if batch.size() == 0:
                break
            
            start_time = time.time()
            output_batch = self.engine.run_batch(batch)
            elapsed = time.time() - start_time
            
            finished_sequences += self._post_process_batch(output_batch, elapsed, is_decode)
             
        self.monitor.print_final_stats()
        return finished_sequences