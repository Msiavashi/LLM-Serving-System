import time
import asyncio  # needed for async loop
from src.sequence import Sequence, Stage
from src.queues import FCFSQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.monitoring.performance_monitor import PerformanceMonitor
from .base_scheduler import BaseScheduler

class BaseFCFSScheduler(BaseScheduler):
    def __init__(self, engine, tokenizer, batch_size):
        self.engine = engine
        self.tokenizer = tokenizer
        self.prefill_queue = FCFSQueue()
        self.decode_queue = FCFSQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.monitor = PerformanceMonitor()

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL, priority=0):
        seq = Sequence(prompt, self.tokenizer, stage, priority=priority)
        if stage == Stage.PREFILL:
            self.prefill_queue.enqueue(seq)
        elif stage == Stage.DECODE:
            self.decode_queue.enqueue(seq)

    def _get_next_batch(self):
        is_decode = not self.decode_queue.is_empty()
        if is_decode:
            batch = self.batch_policy.get_next_batch(self.decode_queue)
        else:
            batch = self.batch_policy.get_next_batch(self.prefill_queue)
        return batch, is_decode

    def _post_process_batch(self, output_batch, elapsed, is_decode):
        current_time = time.time()
        current_batch_latencies = []
        high_priority_latencies = []
        finished_sequences = []

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
            sequences=output_batch.sequences,
            elapsed=elapsed,
            sequence_latencies=current_batch_latencies,
            high_priority_latencies=high_priority_latencies
        )
        return finished_sequences

    def run_loop(self, execute_batch):
        finished_sequences = []
        while not (self.decode_queue.is_empty() and self.prefill_queue.is_empty()):
            batch, is_decode = self._get_next_batch()
            if batch.size() == 0:
                break
            start_time = time.time()
            output_batch = execute_batch(batch)
            elapsed = time.time() - start_time
            finished_sequences += self._post_process_batch(output_batch, elapsed, is_decode)
        self.monitor.print_final_stats()
        return finished_sequences

    async def run_loop_async(self, execute_batch):
        finished_sequences = []
        while not (self.decode_queue.is_empty() and self.prefill_queue.is_empty()):
            batch, is_decode = self._get_next_batch()
            if batch.size() == 0:
                await asyncio.sleep(0.001)
                continue
            start_time = time.time()
            output_batch = await execute_batch(batch)
            elapsed = time.time() - start_time
            finished_sequences += self._post_process_batch(output_batch, elapsed, is_decode)
        self.monitor.print_final_stats()
        return finished_sequences