import torch
import time
from typing import List, Any, Tuple
from .base_engine import BaseEngine
from src.sequence import Sequence

class ModelEngine(BaseEngine):
    """Standard engine for single model execution"""
    def __init__(self, model):
        self.model = model

    def run_batch(self, batch) -> List[Sequence]:
        """Execute a single batch on the model with timing and metrics"""
        
        with torch.no_grad():
            output_batch = self.model(batch=batch, use_cache=True)
            
            return output_batch

    def run_with_timing(self, process_fn: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Run a process with detailed timing information"""
        start_time = time.time()
        
        result = process_fn(*args, **kwargs)
        
        total_duration = time.time() - start_time
        
        return result, total_duration
