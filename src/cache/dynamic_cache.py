import torch
from typing import List
from transformers.cache_utils import DynamicCache

class DynamicCacheEx(DynamicCache):
    def __init__(self):
        super().__init__()
        
    def transfer_layer_to(self, layer_idx, device: torch.device):
        self.key_cache[layer_idx] = k = self.key_cache[layer_idx].to(device)
        self.value_cache[layer_idx] = v = self.value_cache[layer_idx].to(device)
        return k, v
    
    def get_cache_size_at_layer(self, layer_idx: int, unit="mb"):
        return self._convert_size(self._calculate_size([self.key_cache[layer_idx], self.value_cache[layer_idx]]), unit)
        
    def get_cache_size(self, unit="mb"):
        caches = [kv for layer in zip(self.key_cache, self.value_cache) for kv in layer]
        return self._convert_size(self._calculate_size(caches), unit)
    
    @staticmethod
    def _calculate_size(tensors: List[torch.Tensor]) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    
    @staticmethod
    def _convert_size(size: int, unit: str) -> float:
        unit = unit.lower()
        if unit == "bytes":
            return size
        elif unit == "kb":
            return size / 1024
        elif unit == "mb":
            return size / (1024 ** 2)
        elif unit == "gb":
            return size / (1024 ** 3)
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 'bytes', 'KB', 'MB', or 'GB'.")

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

