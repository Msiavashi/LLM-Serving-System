import torch
import time
from typing import Any, Tuple
from .base_engine import BaseEngine

class ModelEngine(BaseEngine):
    """Standard engine for single model execution"""
    def __init__(self, model, cache_provider, **kwargs):
        self.model = model
        self._cache_provider = cache_provider

    def run_batch(self, batch):
        """Execute a single batch on the model with timing and metrics"""
        
        with torch.no_grad():
            return self.model(batch=batch, use_cache=True)

    def run_with_timing(self, process_fn: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Run a process with detailed timing information"""
        start_time = time.time()
        
        result = process_fn(*args, **kwargs)
        
        total_duration = time.time() - start_time
        
        return result, total_duration

    @property
    def cache_provider(self):
        return self._cache_provider

    @cache_provider.setter
    def cache_provider(self, value):
        self._cache_provider = value