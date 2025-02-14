import time
import asyncio
from src.sequence import Stage
from src.queues.fcfs_queue import FCFSQueue
from src.queues.storage.redis_storage import RedisQueueStorage
from src.schedulers.base_fcfs_scheduler import BaseFCFSScheduler 

class AsyncFCFSScheduler(BaseFCFSScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        super().__init__(engine, tokenizer, batch_size)
        # Use Redis storage for prefill queue
        self.prefill_queue = FCFSQueue(RedisQueueStorage("prefill_queue", tokenizer, Stage.PREFILL))
        # Keep in-memory queue for decode
        self.decode_queue = FCFSQueue()

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL):
        raise NotImplementedError

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
            if self.decode_queue.is_empty() and self.prefill_queue.is_empty():
                await asyncio.sleep(0.001)
                continue
            
            batch, is_decode = self._get_next_batch()
                        
            if batch.size() == 0:
                continue
            
            start_time = time.time()
            output_batch = await self.engine.run_batch_async(batch)
            elapsed = time.time() - start_time
            
            finished_sequences += self._post_process_batch(output_batch, elapsed, is_decode)

