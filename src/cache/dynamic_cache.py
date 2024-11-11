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
        Merges KV caches with GPU optimization.
        """
        merged_cache = cls()
        if not caches or caches[0] is None:
            return merged_cache

        # Pre-transfer caches to GPU once
        for cache in caches:
            for layer_idx in range(len(cache)):
                if cache.key_cache[layer_idx].device.type != 'cuda':
                    cache.key_cache[layer_idx] = cache.key_cache[layer_idx].cuda()
                if cache.value_cache[layer_idx].device.type != 'cuda':
                    cache.value_cache[layer_idx] = cache.value_cache[layer_idx].cuda()

        for layer_idx in range(len(caches[0])):
            # Get max lengths once per layer
            layer_keys_list = [cache.key_cache[layer_idx] for cache in caches]
            layer_values_list = [cache.value_cache[layer_idx] for cache in caches]
            
            max_key_len = max(tensor.shape[2] for tensor in layer_keys_list)
            max_value_len = max(tensor.shape[2] for tensor in layer_values_list)
            # Batch padding operations on GPU
            padded_keys = torch.cat([
                torch.nn.functional.pad(tensor, (0, 0, 0, max_key_len - tensor.shape[2]))
                for tensor in layer_keys_list
            ], dim=0)
            
            padded_values = torch.cat([
                torch.nn.functional.pad(tensor, (0, 0, 0, max_value_len - tensor.shape[2]))
                for tensor in layer_values_list
            ], dim=0)
            
            merged_cache.update(padded_keys, padded_values, layer_idx)
            
        return merged_cache


    # @classmethod
    # def merge_kv_caches(cls, caches: List["DynamicCacheEx"]) -> "DynamicCacheEx":
    #     """
    #     Merges a list of MyDynamicCache instances into a single MyDynamicCache instance.
        
    #     Parameters:
    #         caches (List[MyDynamicCache]): A list of MyDynamicCache instances to be merged.
        
    #     Returns:
    #         MyDynamicCache: A new MyDynamicCache instance containing the merged KV caches.
    #     """
    #     merged_cache = cls()
    #     if caches or caches[0] is not None:
    #         for layer_idx in range(len(caches[0])):
    #             layer_keys = torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=0)
    #             layer_values = torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=0)
    #             merged_cache.update(layer_keys, layer_values, layer_idx)
    #     return merged_cache