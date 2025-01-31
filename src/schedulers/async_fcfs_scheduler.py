import time
import asyncio

from src.sequence import Stage
from src.queues.fcfs_queue import FCFSQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.monitoring.performance_monitor import PerformanceMonitor
from .base_scheduler import BaseScheduler
from src.queues.storage.redis_storage import RedisQueueStorage

class AsyncFCFSScheduler(BaseScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        self.engine = engine
        self.tokenizer = tokenizer
        # Use Redis storage for prefill queue
        self.prefill_queue = FCFSQueue(RedisQueueStorage("prefill_queue", tokenizer, Stage.PREFILL))
        # Keep in-memory queue for decode
        self.decode_queue = FCFSQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.monitor = PerformanceMonitor()

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL):
        pass

    async def run_scheduler(self):
        finished_sequences = []
        
        while True:  # Run forever
            if self.decode_queue.is_empty() and self.prefill_queue.is_empty():
                await asyncio.sleep(0.001)
                continue
                
            is_decode = not self.decode_queue.is_empty()
            
            if is_decode:
                batch = self.batch_policy.get_next_batch(self.decode_queue)
            else:
                batch = self.batch_policy.get_next_batch(self.prefill_queue)
             
            if batch.size() == 0:
                continue
            
            start_time = time.time()
            # Use the async version of run_batch
            output_batch = await self.engine.run_batch_async(batch)
            elapsed = time.time() - start_time
            
            current_time = time.time()
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
                    self.decode_queue.enqueue(seq)
            
            self.monitor.record_batch(
                is_decode=is_decode,
                tokens_generated=len(output_batch.sequences),
                elapsed=elapsed,
                sequence_latencies=current_batch_latencies,
                high_priority_latencies=high_priority_latencies
            )
            
            # Don't print stats continuously
            # if len(finished_sequences) % 100 == 0:
            #     self.monitor.print_final_stats()

