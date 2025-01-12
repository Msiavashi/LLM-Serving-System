import torch
import time
from typing import List, Any, Tuple
from .base_engine import BaseEngine
from src.batching.batch import Batch

class AsyncModelEngine(BaseEngine):
    """Asynchronous engine for single model execution"""
    def __init__(self, model):
        self.model = model

    def run_batch(self, batch: Batch) -> Batch:
        """Execute a single batch on the model"""
        with torch.no_grad():
            # Process the batch synchronously since torch operations are sync
            output = self.model(batch=batch, use_cache=True)
            return output

    async def run_batch_async(self, batch: Batch) -> Batch:
        """Async wrapper for run_batch"""
        # Use asyncio.to_thread if the operation is CPU-bound
        return self.run_batch(batch)

    async def run_with_timing(self, process_fn: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Run a process with detailed timing information"""
        start_time = time.time()
        
        result = process_fn(*args, **kwargs)
        
        total_duration = time.time() - start_time
        
        return result, total_duration
