from typing import Any, Dict, List, Optional, Tuple
from .dynamic_cache import DynamicCacheEx as DynamicCache
import torch

class UnifiedDynamicCache(DynamicCache):
    
    def __init__(self, caches: List[DynamicCache] = None):
        self.caches: List[DynamicCache] = caches if caches is not None else []
        super().__init__()
        
    def split_kv_cache(self):
        return self.caches
    
    def _update_single_cache(self, cache, key_state, value_state, layer_idx, cache_kwargs):
        return cache.update(key_state, value_state, layer_idx, cache_kwargs)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            # Update caches and track max length
            max_len = 0
            updated_keys = []
            updated_values = []
            
            for i, cache in enumerate(self.caches):
                k, v = cache.update(key_states[i:i+1], value_states[i:i+1], layer_idx, cache_kwargs)
                updated_keys.append(k)
                updated_values.append(v)
                max_len = max(max_len, k.shape[2])  # Update max_len based on sequence length (dim 2)
            
            # Ensure all keys and values have the same sequence length by padding
            for i in range(len(updated_keys)):
                if updated_keys[i].shape[2] < max_len:
                    pad_size = max_len - updated_keys[i].shape[2]
                    updated_keys[i] = torch.nn.functional.pad(updated_keys[i], (0, 0, 0, pad_size))
                    updated_values[i] = torch.nn.functional.pad(updated_values[i], (0, 0, 0, pad_size))
            
            return torch.cat(updated_keys, dim=0), torch.cat(updated_values, dim=0)

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