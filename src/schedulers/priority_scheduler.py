import time

from .base_priority_scheduler import BasePriorityScheduler


class PriorityScheduler(BasePriorityScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        super().__init__(engine, tokenizer, batch_size)

    def run_scheduler(self):
        finished_sequences = []

        while not (self.ls_decode_queue.is_empty() and self.ls_prefill_queue.is_empty() and
                   self.non_ls_decode_queue.is_empty() and self.non_ls_prefill_queue.is_empty()):

            batch = self._get_next_batch()

            if batch and batch.size() >= 0:
                finished = self._process_batch(batch)
                finished_sequences.extend(finished)
            else:
                break

        self.monitor.print_final_stats()
        return finished_sequences
