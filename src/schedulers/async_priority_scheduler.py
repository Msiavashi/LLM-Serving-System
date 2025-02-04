import time
import asyncio

from src.sequence import Sequence, Stage
from src.queues.fcfs_queue import FCFSQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.monitoring.performance_monitor import PerformanceMonitor
from .base_priority_scheduler import BasePriorityScheduler
from src.queues.storage.redis_storage import RedisQueueStorage

class AsyncPriorityScheduler(BasePriorityScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        super().__init__(engine, tokenizer, batch_size)
        # Use Redis storage for prefill queues
        self.non_ls_prefill_queue = FCFSQueue(RedisQueueStorage("non_ls_prefill_queue", tokenizer, Stage.PREFILL))
        self.ls_prefill_queue = FCFSQueue(RedisQueueStorage("ls_prefill_queue", tokenizer, Stage.PREFILL))
        # Keep in-memory queues for decode
        self.non_ls_decode_queue = FCFSQueue()
        self.ls_decode_queue = FCFSQueue()

    async def _process_batch(self, batch):
        is_decode = batch.is_decode()
        start_time = time.time()
        output_batch = await self.engine.run_batch_async(batch)
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
                print("--" * 40)
                print(f"\nSequence ID: {seq.sequence_id}, Turnaround Time: {seq.finish_time - seq.arrival_time}, Priority: {seq.priority}\n")
                print("--" * 40)
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

    async def run_scheduler(self):
        finished_sequences = []

        async def print_stats_periodically():
            while True:
                await asyncio.sleep(15)
                print("\n" + "="*80)
                self.monitor.print_final_stats()
                print("="*80 + "\n")

        asyncio.create_task(print_stats_periodically())

        while True:  # Run forever
            if (self.ls_decode_queue.is_empty() and self.ls_prefill_queue.is_empty() and
                self.non_ls_decode_queue.is_empty() and self.non_ls_prefill_queue.is_empty()):
                await asyncio.sleep(0.001)
                continue

            batch = self._get_next_batch()

            if batch and batch.size() > 0:
                finished = await self._process_batch(batch)
                finished_sequences.extend(finished)
            else:
                continue
