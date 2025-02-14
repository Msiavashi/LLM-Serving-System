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
        async def print_stats_periodically():
            while True:
                await asyncio.sleep(15)
                print("\n" + "="*80)
                self.monitor.print_final_stats()
                print("="*80 + "\n")
        asyncio.create_task(print_stats_periodically())
        return await self.run_loop_async(lambda batch: self.engine.run_batch_async(batch))

