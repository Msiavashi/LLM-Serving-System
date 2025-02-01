import time

from src.sequence import Sequence, Stage
from src.queues import FCFSQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.monitoring.performance_monitor import PerformanceMonitor
from .base_scheduler import BaseScheduler
from .utils import log_queue_sizes

class PriorityScheduler(BaseScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        self.engine = engine
        self.tokenizer = tokenizer
        self.non_ls_prefill_queue = FCFSQueue()
        self.ls_prefill_queue = FCFSQueue() # queue for latency sensitive requests
        self.non_ls_decode_queue = FCFSQueue()
        self.ls_decode_queue = FCFSQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.monitor = PerformanceMonitor()

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL, priority=0):
        seq = Sequence(prompt, self.tokenizer, stage, priority=priority)
        if stage == Stage.PREFILL:
            if priority == 1:
                self.ls_prefill_queue.enqueue(seq)
            else:
                self.non_ls_prefill_queue.enqueue(seq)
        elif stage == Stage.DECODE:
            if priority == 1:
                self.ls_decode_queue.enqueue(seq)
            else:
                self.non_ls_decode_queue.enqueue(seq)

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
            tokens_generated=len(output_batch.sequences),
            elapsed=elapsed,
            sequence_latencies=current_batch_latencies,
            high_priority_latencies=high_priority_latencies
        )
        
        return finished_sequences

    def run_scheduler(self):
        finished_sequences = []
        
        while not (self.ls_decode_queue.is_empty() and self.ls_prefill_queue.is_empty() and
                  self.non_ls_decode_queue.is_empty() and self.non_ls_prefill_queue.is_empty()):
            
            batch = None
            
            # First priority: LS decode queue if it has enough sequences
            if self.ls_decode_queue.size() >= self.batch_policy.batch_size:
                batch = self.batch_policy.get_next_batch(self.ls_decode_queue)
            
            # Second priority: LS prefill if LS decode doesn't have enough sequences
            elif not self.ls_prefill_queue.is_empty():
                batch = self.batch_policy.get_next_batch(self.ls_prefill_queue)
                if batch.size() > 0:
                    finished = self._process_batch(batch)  # Remove current_time parameter
                    finished_sequences.extend(finished)
                    # Try to process LS decode queue again
                    if not self.ls_decode_queue.is_empty():
                        batch = self.batch_policy.get_next_batch(self.ls_decode_queue)
                    else:
                        continue
            
            # Third priority: Process remaining LS decode sequences even if less than batch_size
            elif not self.ls_decode_queue.is_empty():
                batch = self.batch_policy.get_next_batch(self.ls_decode_queue)
            
            # Fourth priority: Non-LS decode queue
            elif not self.non_ls_decode_queue.is_empty():
                batch = self.batch_policy.get_next_batch(self.non_ls_decode_queue)
            
            # Fifth priority: Non-LS prefill queue
            elif not self.non_ls_prefill_queue.is_empty():
                batch = self.batch_policy.get_next_batch(self.non_ls_prefill_queue)
            
            if batch and batch.size() > 0:
                finished = self._process_batch(batch)  # Remove current_time parameter
                finished_sequences.extend(finished)
            else:
                break
        
        # self.monitor.print_final_stats()
        # return finished_sequences