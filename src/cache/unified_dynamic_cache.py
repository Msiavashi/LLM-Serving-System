from typing import Any, Dict, List, Optional, Tuple
from .dynamic_cache import DynamicCacheEx as DynamicCache
import torch

class UnifiedDynamicCache(DynamicCache):
    
    def __init__(self, caches: List[DynamicCache] = None):
        self.caches: List[DynamicCache] = caches if caches is not None else []
        super().__init__()
        self.max_len = 0
        
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
            max_len = 0
            
            for i, cache in enumerate(self.caches):
                keys, values = cache.update(key_states[i], value_states[i], layer_idx, cache_kwargs)
                key_list.append(keys)
                value_list.append(values)
                max_len = max(max_len, keys.shape[1])
            self.max_len = max_len
            
            # Efficient padding
            padded_key_list = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.shape[1])) if tensor.shape[1] < max_len else tensor for tensor in key_list]
            padded_value_list = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.shape[1])) if tensor.shape[1] < max_len else tensor for tensor in value_list]
            
            merged_key_states = torch.stack(padded_key_list)
            merged_value_states = torch.stack(padded_value_list)
            
            return merged_key_states, merged_value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        seq_lengths = [cache.get_seq_length(layer_idx) for cache in self.caches]
        return min(seq_lengths) if seq_lengths else 0

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        # Get the minimum usable length across all caches for safety
        usable_lengths = [cache.get_usable_length(new_seq_length, layer_idx) for cache in self.caches]
        return min(usable_lengths) if usable_lengths else new_seq_length
    
    def get_cache_size(self, unit="mb"):
        return sum(cache.get_cache_size(unit) for cache in self.caches)
    
    def get_cache_size_at_layer(self, layer_idx, unit="mb"):
         return sum(cache.get_cache_size_at_layer(layer_idx, unit) for cache in self.caches)
     
    def transfer_layer_to(self, layer_idx, device):
         return [cache.transfer_layer_to(layer_idx, device) for cache in self.caches]