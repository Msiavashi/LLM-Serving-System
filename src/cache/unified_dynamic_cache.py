from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from src.cache.compression import compress, decompress
from transformers.cache_utils import DynamicCache


class UnifiedDynamicCache(DynamicCache):
    def __init__(self, caches: Optional[List[DynamicCache]] = None):
        super().__init__()
        self.caches: List[DynamicCache] = caches if caches is not None else []

    def split_kv_cache(self) -> List[DynamicCache]:
        return self.caches

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.caches) == 1:
            return self.caches[0].update(key_states, value_states, layer_idx, cache_kwargs)

        orig_k_dtype = key_states.dtype
        orig_v_dtype = value_states.dtype

        # Collect updates
        updated = [
            cache.update(key_states[i : i + 1], value_states[i : i + 1], layer_idx, cache_kwargs)
            for i, cache in enumerate(self.caches)
        ]
        updated_keys, updated_values = zip(*updated)

        # Pad to common length (cannot be in-place)
        max_len = max(k.shape[2] for k in updated_keys)
        if any(k.shape[2] != max_len for k in updated_keys):
            padded_keys = []
            padded_values = []
            for k, v in zip(updated_keys, updated_values):
                pad_size = max_len - k.shape[2]
                if pad_size > 0:
                    k = F.pad(k, (0, 0, 0, pad_size))
                    v = F.pad(v, (0, 0, 0, pad_size))
                padded_keys.append(k)
                padded_values.append(v)
            updated_keys, updated_values = padded_keys, padded_values
        else:
            updated_keys = list(updated_keys)
            updated_values = list(updated_values)

        # TODO: Compress / Decompress (single pass) This is an example here. An optional compression must be implemented to compress KV cache on demand.
        comp_k, comp_v, k_meta, v_meta = compress(updated_keys, updated_values)
        dec_k, dec_v = decompress(comp_k, comp_v, k_meta, v_meta)

        # Final cast only once
        dec_k = [k.to(orig_k_dtype) for k in dec_k]
        dec_v = [v.to(orig_v_dtype) for v in dec_v]

        return torch.cat(dec_k, dim=0), torch.cat(dec_v, dim=0)

    def get_key_cache(self, layer_idx: Optional[int] = 0) -> List[torch.Tensor]:
        return [cache.key_cache[layer_idx] for cache in self.caches]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        seq_lengths = [cache.get_seq_length(layer_idx) for cache in self.caches]
        return min(seq_lengths) if seq_lengths else 0

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        usable = [cache.get_usable_length(new_seq_length, layer_idx) for cache in self.caches]
        return min(usable) if usable else new_seq_length

    def get_cache_size(self, unit="mb"):
        return sum(cache.get_cache_size(unit) for cache in self.caches)

    def get_cache_size_at_layer(self, layer_idx, unit="mb"):
        return sum(cache.get_cache_size_at_layer(layer_idx, unit) for cache in self.caches)

    def transfer_layer_to(self, layer_idx, device):
        return [cache.transfer_layer_to(layer_idx, device) for cache in self.caches]

    def __len__(self):
        return len(self.caches[0]) if self.caches else 0