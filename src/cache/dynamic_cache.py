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
            List[DynamicCacheEx]: A list of DynamicCacheEx instances, each containing one sequence's KV cache.
        """
        split_caches = [DynamicCacheEx() for _ in range(batch_size)]

        for layer_idx in range(len(self)):
            self._split_layer_cache(layer_idx, split_caches, batch_size)

        return split_caches

    def _split_layer_cache(self, layer_idx: int, split_caches: List["DynamicCacheEx"], batch_size: int):
        """
        Splits the KV cache for a specific layer and updates the split caches.

        Parameters:
            layer_idx (int): The layer index to split.
            split_caches (List[DynamicCacheEx]): List of split caches to update.
            batch_size (int): The batch size representing the number of sequences.
        """
        for i in range(batch_size):
            key_split = self.key_cache[layer_idx][i:i+1]  # Select a single sequence for keys
            value_split = self.value_cache[layer_idx][i:i+1]  # Select a single sequence for values
            split_caches[i].update(key_split, value_split, layer_idx)

    @classmethod
    def merge_kv_caches(cls, caches: List["DynamicCacheEx"], layer_idx: int = None) -> "DynamicCacheEx":
        """
        Merges multiple KV caches into a single cache.

        Parameters:
            caches (List[DynamicCacheEx]): The list of KV caches to merge.
            layer_idx (int, optional): The specific layer to merge. Merges all layers if None.

        Returns:
            DynamicCacheEx: A new DynamicCacheEx instance with merged KV caches.
        """
        merged_cache = cls()

        if not caches or caches[0] is None:
            return merged_cache

        layers_to_merge = range(len(caches[0])) if layer_idx is None else [layer_idx]

        for idx in layers_to_merge:
            cls._merge_layer_cache(idx, caches, merged_cache)

        return merged_cache

    @staticmethod
    def _merge_layer_cache(layer_idx: int, caches: List["DynamicCacheEx"], merged_cache: "DynamicCacheEx"):
        """
        Merges KV caches for a specific layer.

        Parameters:
            layer_idx (int): The layer index to merge.
            caches (List[DynamicCacheEx]): The list of KV caches to merge.
            merged_cache (DynamicCacheEx): The merged cache to update.
        """
        layer_keys = [cache.key_cache[layer_idx] for cache in caches]
        layer_values = [cache.value_cache[layer_idx] for cache in caches]

        max_len = max(key.shape[2] for key in layer_keys)

        padded_keys = DynamicCacheEx._pad_to_max_length(layer_keys, max_len)
        padded_values = DynamicCacheEx._pad_to_max_length(layer_values, max_len)

        keys = torch.stack(padded_keys)
        values = torch.stack(padded_values)

        merged_cache.update(keys.reshape(-1, *keys.shape[2:]), values.reshape(-1, *values.shape[2:]), layer_idx)

    @staticmethod
    def _pad_to_max_length(tensors: List[torch.Tensor], max_len: int) -> List[torch.Tensor]:
        """
        Pads a list of tensors to the maximum length along the last dimension.

        Parameters:
            tensors (List[torch.Tensor]): The list of tensors to pad.
            max_len (int): The maximum length to pad to.

        Returns:
            List[torch.Tensor]: The padded tensors.
        """
        return [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.shape[2])) for tensor in tensors]

    @classmethod
    def merge_single_layer(cls, caches: List["DynamicCacheEx"], layer_idx: int) -> "DynamicCacheEx":
        """
        Merges a single layer from multiple caches into a single cache object.

        Parameters:
            caches (List[DynamicCacheEx]): The list of caches to merge from.
            layer_idx (int): The layer index to merge.

        Returns:
            DynamicCacheEx: A new DynamicCacheEx instance containing only the merged layer.
        """
        merged_cache = cls()

        layer_keys = [cache.key_cache[layer_idx] for cache in caches]
        layer_values = [cache.value_cache[layer_idx] for cache in caches]

        max_len = max(key.shape[2] for key in layer_keys)

        padded_keys = cls._pad_to_max_length(layer_keys, max_len)
        padded_values = cls._pad_to_max_length(layer_values, max_len)

        keys = torch.stack(padded_keys)
        values = torch.stack(padded_values)

        merged_cache.update(keys.reshape(-1, *keys.shape[2:]), values.reshape(-1, *values.shape[2:]), layer_idx)
        return merged_cache


    def split_layer_to_caches(self, layer_idx: int, caches: List["DynamicCacheEx"]):
        """
        Splits a cache object for a specific layer and updates the list of caches at the given layer index.

        The `self` object contains the batch dimension.

        Parameters:
            layer_idx (int): The layer index to split.
            caches (List[DynamicCacheEx]): The list of caches to update.
        """
        batch_size = len(caches)
        for i in range(batch_size):
            key_split = self.key_cache[layer_idx][i:i+1]
            value_split = self.value_cache[layer_idx][i:i+1]
            caches[i].update(key_split, value_split, layer_idx)
        return caches
