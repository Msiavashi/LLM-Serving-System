import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import DynamicCache

class DynamicCacheEx(DynamicCache):
    def __init__(self):
        super().__init__()

    def split_kv_cache(self, batch_size: int) -> List["DynamicCacheEx"]:
        split_caches = [DynamicCacheEx() for _ in range(batch_size)]
        if not self.key_cache:
            return split_caches

        for layer_idx in range(len(self)):
            keys = self.key_cache[layer_idx].chunk(batch_size, dim=0)
            values = self.value_cache[layer_idx].chunk(batch_size, dim=0)
            for i in range(batch_size):
                split_caches[i].update(keys[i], values[i], layer_idx)

        return split_caches

    @classmethod
    def merge_kv_caches(cls, caches: List["DynamicCacheEx"], layer_idx: int = None) -> "DynamicCacheEx":
        merged_cache = cls()
        if not caches or not caches[0] or not caches[0].key_cache:
            return merged_cache

        layers_to_merge = range(len(caches[0]))
        for idx in layers_to_merge:
            cls._merge_layer_cache(idx, caches, merged_cache, pad=True)

        return merged_cache
    
    @staticmethod
    def _merge_layer_cache(layer_idx: int, caches: List["DynamicCacheEx"], merged_cache: "DynamicCacheEx", pad: bool = True):
        layer_keys = [cache.key_cache[layer_idx] for cache in caches]
        layer_values = [cache.value_cache[layer_idx] for cache in caches]

        if pad:
            max_len = max(key.shape[2] for key in layer_keys)
            padded_keys = DynamicCacheEx._pad_to_max_length(layer_keys, max_len)
            padded_values = DynamicCacheEx._pad_to_max_length(layer_values, max_len)
        else:
            padded_keys = layer_keys
            padded_values = layer_values

        keys = torch.stack(padded_keys)
        values = torch.stack(padded_values)

        merged_cache.update(keys.reshape(-1, *keys.shape[2:]), values.reshape(-1, *values.shape[2:]), layer_idx)
        
    @staticmethod
    def _pad_to_max_length(tensors: List[torch.Tensor], max_len: int) -> List[torch.Tensor]:
        return [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.shape[2])) for tensor in tensors]

class UnifiedDynamicCache(DynamicCacheEx):
    
    
    def __init__(self, caches: List[DynamicCacheEx] = None):
        self.caches: List[DynamicCacheEx] = caches if caches is not None else []
        super().__init__()
        
    def split_kv_cache(self):
        return self.caches
    
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            key_list = []
            value_list = []
            for i, cache in enumerate(self.caches):
                keys, values = cache.update(key_states[i], value_states[i], layer_idx, cache_kwargs)
                key_list.append(keys)
                value_list.append(values)
            
            merged_key_states = torch.stack(key_list)
            merged_value_states = torch.stack(value_list)

            return merged_key_states, merged_value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        seq_lengths = [cache.get_seq_length(layer_idx) for cache in self.caches]
        return min(seq_lengths) if seq_lengths else 0