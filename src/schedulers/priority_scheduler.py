import time

from src.queues import FCFSQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.monitoring.performance_monitor import PerformanceMonitor
from .base_priority_scheduler import BasePriorityScheduler

class PriorityScheduler(BasePriorityScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        self.engine = engine
        self.tokenizer = tokenizer
        self.non_ls_prefill_queue = FCFSQueue()
        self.ls_prefill_queue = FCFSQueue() # queue for latency sensitive requests
        self.non_ls_decode_queue = FCFSQueue()
        self.ls_decode_queue = FCFSQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.monitor = PerformanceMonitor()

    def _process_batch(self, batch):
        is_decode = batch.is_decode()
        start_time = time.time()
        output_batch = self.engine.run_batch(batch)
        elapsed = time.time() - start_time
         
        current_time = time.time()
        
        finished_sequences = []
        current_batch_latencies = []
        high_priority_latencies = []
         
        for seq in output_batch.sequences:
            current_latency = current_time - seq.previous_token_time
            current_batch_latencies.append(current_latency)
            if seq.priority == 1:
                high_priority_latencies.append(current_latency)
            seq.previous_token_time = current_time
            seq.sampling_metadata.current_token_count += 1
            
            if seq.sampling_metadata.current_token_count >= seq.sampling_metadata.max_sequence_length:
                seq.finish_time = current_time
                finished_sequences.append(seq)
                del seq.kv_cache
            else:
                if seq.priority == 1:
                    self.ls_decode_queue.enqueue(seq)
                else:
                    self.non_ls_decode_queue.enqueue(seq)
        
        self.monitor.record_batch(
            is_decode=is_decode,
            sequences=output_batch.sequences,
            elapsed=elapsed,
            sequence_latencies=current_batch_latencies,
            high_priority_latencies=high_priority_latencies
        )
        
        return finished_sequences

    
    def run_scheduler(self):
        finished_sequences = []

        while not (self.ls_decode_queue.is_empty() and self.ls_prefill_queue.is_empty() and
                   self.non_ls_decode_queue.is_empty() and self.non_ls_prefill_queue.is_empty()):

            # Print number of queued tokens from the model.
            # print("Queued tokens:", self.engine.model.model.has_queued_tokens())

            batch = self._get_next_batch()

            if batch and batch.size() >= 0:
                finished = self._process_batch(batch)
                finished_sequences.extend(finished)
            else:
                break

        self.monitor.print_final_stats()
        return finished_sequences