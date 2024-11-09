import torch
from typing import List
from transformers.cache_utils import DynamicCache

class DynamicCacheEx(DynamicCache):
    def __init__(self):
        super().__init__()

    def split_kv_cache(self, batch_size: int) -> List["DynamicCacheEx"]:
        """
        Splits the KV cache into multiple caches, each corresponding to a separate sequence in the batch.
        
        Parameters:
            batch_size (int): The batch size, representing the number of sequences.
        
        Returns:
            List[MyDynamicCache]: A list of MyDynamicCache instances, each containing one sequence's KV cache.
        """
        split_caches = [DynamicCacheEx() for _ in range(batch_size)]
        
        for layer_idx in range(len(self)):
            for i in range(batch_size):
                key_split = self.key_cache[layer_idx][i:i+1]  # Select a single sequence for keys
                value_split = self.value_cache[layer_idx][i:i+1]  # Select a single sequence for values
                split_caches[i].update(key_split, value_split, layer_idx)
        
        return split_caches

    @classmethod
    def merge_kv_caches(cls, caches: List["DynamicCacheEx"]) -> "DynamicCacheEx":
        """
        Merges a list of MyDynamicCache instances into a single MyDynamicCache instance.
        
        Parameters:
            caches (List[MyDynamicCache]): A list of MyDynamicCache instances to be merged.
        
        Returns:
            MyDynamicCache: A new MyDynamicCache instance containing the merged KV caches.
        """
        merged_cache = cls()
        if caches and caches[0] is not None:
            for layer_idx in range(len(caches[0])):
                layer_keys = torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=0)
                layer_values = torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=0)
                merged_cache.update(layer_keys, layer_values, layer_idx)
        return merged_cache
