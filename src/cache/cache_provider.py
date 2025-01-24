import torch
from transformers.cache_utils import StaticCache
from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
import logging

class CacheProvider:
    def __init__(self, model_config, cache_type="dynamic"):
        self.model_config = model_config
        self.cache_type = cache_type

    def create_sequence_cache(self, batch_size=None, max_sequence_length=None):
        if self.cache_type == "static":
            if batch_size is None or max_sequence_length is None:
                raise ValueError("batch_size and max_sequence_length must be provided for static cache")
            return self._create_static_cache(batch_size, max_sequence_length)
        
        elif self.cache_type == "dynamic":
            if batch_size is not None or max_sequence_length is not None:
                raise ValueError("batch_size and max_sequence_length are not used for dynamic cache")
            return self._create_dynamic_cache()
        
        else:
            raise ValueError(f"Unsupported cache type: {self.cache_type}")

    def _create_static_cache(self, batch_size, max_sequence_length):
        return StaticCache(
            self.model_config, 
            batch_size=batch_size,
            max_cache_len=max_sequence_length,
            device=torch.device("cuda:1"),
            dtype=torch.half
        )

    def _create_dynamic_cache(self):
        return DynamicCache()
