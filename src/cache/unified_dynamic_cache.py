from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .lmcache_wrapper import LMCacheWrapper


class UnifiedDynamicCache(DynamicCache):

    def __init__(
        self,
        caches: Optional[List[DynamicCache]] = None,
        *,
        use_lmcache: bool = False,
        lmcache: Optional[LMCacheWrapper] = None,
    ):
        super().__init__()
        self.caches: List[DynamicCache] = caches if caches is not None else []
        self.use_lmcache = use_lmcache
        self.lmcache = lmcache

    def split_kv_cache(self) -> List[DynamicCache]:
        return self.caches

    def store(self, tokens_list: List[torch.Tensor]) -> None:
        """Store caches for each sequence in *tokens_list* to LMCache."""
        if not (self.use_lmcache and self.lmcache):
            return
        for tokens, cache in zip(tokens_list, self.caches):
            kv = [(cache.key_cache[i], cache.value_cache[i]) for i in range(len(cache))]
            self.lmcache.store(tokens, kv)

    def retrieve(self, tokens_list: List[torch.Tensor]) -> None:
        """Retrieve caches from LMCache for each sequence in *tokens_list*."""
        if not (self.use_lmcache and self.lmcache):
            return
        retrieved = []
        for tokens in tokens_list:
            kv = self.lmcache.retrieve(tokens)
            if kv:
                new_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(kv):
                    new_cache.update(k, v, layer_idx)
                retrieved.append(new_cache)
            else:
                retrieved.append(DynamicCache())
        self.caches = retrieved

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collect updated keys/values from each sub-cache
        updated_keys = []
        updated_values = []
        for i, cache in enumerate(self.caches):
            k, v = cache.update(
                key_states[i : i + 1], 
                value_states[i : i + 1], 
                layer_idx, 
                cache_kwargs
            )
            updated_keys.append(k)
            updated_values.append(v)

        # Compute max sequence length across all updated keys
        max_len = max(k.shape[2] for k in updated_keys)

        # Pad keys/values to ensure uniform sequence length
        for i, (k, v) in enumerate(zip(updated_keys, updated_values)):
            seq_len = k.shape[2]
            if seq_len < max_len:
                pad_size = max_len - seq_len
                updated_keys[i] = F.pad(k, (0, 0, 0, pad_size))
                updated_values[i] = F.pad(v, (0, 0, 0, pad_size))

        # Concatenate along the batch dimension (dim=0)
        return torch.cat(updated_keys, dim=0), torch.cat(updated_values, dim=0)


    def get_key_cache(self, layer_idx: Optional[int] = 0) -> List[torch.Tensor]:
        # Collect key caches from each sub-cache
        key_caches = [cache.key_cache[layer_idx] for cache in self.caches]
        return key_caches

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
     
    def __len__(self):
        return len(self.caches[0]) if self.caches else 0
