import torch
import time
from typing import List, Any, Tuple
from datetime import datetime
from .base_engine import BaseEngine
from src.sequence import Sequence

class ModelEngine(BaseEngine):
    """Standard engine for single model execution"""
    def __init__(self, model):
        self.model = model

    def run_batch(self, batch) -> List[Sequence]:
        """Execute a single batch on the model with timing and metrics"""
        start_time = time.time()
        start_timestamp = datetime.now()
        
        print(f"\nBatch execution starting at: {start_timestamp}")
        print(f"Model device: {self.model.device}")
        
        with torch.no_grad():
            output_batch = self.model(batch=batch, use_cache=True)
            tokens_generated = len(output_batch.sequences)
            elapsed = time.time() - start_time
            
            # Print execution metrics
            print(f"Batch finished at: {datetime.now()}")
            print(f"Throughput = {tokens_generated/elapsed:.2f} tokens/sec")
            print(f"Batch size = {tokens_generated}")
            print(f"Elapsed time = {elapsed:.2f} sec")
            
            return output_batch.sequences

    def run_with_timing(self, process_fn: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Run a process with detailed timing information"""
        start_time = time.time()
        start_timestamp = datetime.now()
        
        print(f"\nProcess execution starting at: {start_timestamp}")
        
        result = process_fn(*args, **kwargs)
        
        total_duration = time.time() - start_time
        print(f"\nProcess finished at: {datetime.now()}")
        print(f"Total execution time: {total_duration:.2f} seconds")
        
        return result, total_duration
