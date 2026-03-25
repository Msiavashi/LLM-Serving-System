"""SequenceCacheManager: Per-sequence KV cache using HF-native cache types.

Works with any HuggingFace cache implementation (DynamicCache, StaticCache,
QuantizedCache, etc.) via composition. Handles:
- Per-sequence cache storage and lifecycle
- Batch assembly (pad + merge) before forward pass
- Batch disassembly (split) after forward pass
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


class SequenceCacheManager:
    """Manages per-sequence KV caches using HF-native cache types."""

    def __init__(self, cache_type: str = "dynamic"):
        self.cache_type = cache_type
        self.per_sequence_caches: Dict[int, DynamicCache] = {}

    def create_cache(self) -> DynamicCache:
        """Create a new HF-native cache instance."""
        if self.cache_type == "dynamic":
            return DynamicCache()
        else:
            # Extensible: add StaticCache, QuantizedCache, etc.
            return DynamicCache()

    def get_or_create(self, seq_id: int) -> DynamicCache:
        """Get existing cache for a sequence, or create a new one."""
        if seq_id not in self.per_sequence_caches:
            self.per_sequence_caches[seq_id] = self.create_cache()
        return self.per_sequence_caches[seq_id]

    def remove(self, seq_id: int):
        """Remove a sequence's cache (on completion)."""
        self.per_sequence_caches.pop(seq_id, None)

    def assemble_batch_cache(self, sequences: list) -> Optional[DynamicCache]:
        """Assemble per-sequence caches into a single batch cache for forward pass.

        Handles variable-length sequences by padding to common length.
        Returns None if no sequences have caches (first prefill).
        """
        caches = []
        for seq in sequences:
            cache = self.per_sequence_caches.get(seq.sequence_id)
            if cache is None or len(cache.key_cache) == 0:
                return None  # At least one sequence has no cache → prefill mode
            caches.append(cache)

        if not caches:
            return None

        # All caches should have the same number of layers
        num_layers = len(caches[0].key_cache)

        # Check if all sequences have the same cache length (common case in decode)
        lengths = [c.get_seq_length() for c in caches]
        all_same_length = len(set(lengths)) == 1

        # Build merged cache
        merged = DynamicCache()
        for layer_idx in range(num_layers):
            keys = [c.key_cache[layer_idx] for c in caches]
            values = [c.value_cache[layer_idx] for c in caches]

            if all_same_length:
                # Fast path: just concatenate along batch dimension
                merged_k = torch.cat(keys, dim=0)
                merged_v = torch.cat(values, dim=0)
            else:
                # Pad to common length then concatenate
                max_len = max(k.shape[2] for k in keys)
                padded_keys = []
                padded_values = []
                for k, v in zip(keys, values):
                    pad_size = max_len - k.shape[2]
                    if pad_size > 0:
                        k = F.pad(k, (0, 0, 0, pad_size))
                        v = F.pad(v, (0, 0, 0, pad_size))
                    padded_keys.append(k)
                    padded_values.append(v)
                merged_k = torch.cat(padded_keys, dim=0)
                merged_v = torch.cat(padded_values, dim=0)

            merged.key_cache.append(merged_k)
            merged.value_cache.append(merged_v)

        # Set _seen_tokens to match
        merged._seen_tokens = max(lengths)
        return merged

    def disassemble_batch_cache(self, batch_cache: DynamicCache, sequences: list):
        """Split a batch cache back into per-sequence caches after forward pass."""
        if batch_cache is None:
            return

        batch_size = len(sequences)
        num_layers = len(batch_cache.key_cache)

        for i, seq in enumerate(sequences):
            cache = DynamicCache()
            for layer_idx in range(num_layers):
                k = batch_cache.key_cache[layer_idx][i:i+1]
                v = batch_cache.value_cache[layer_idx][i:i+1]
                cache.key_cache.append(k)
                cache.value_cache.append(v)
            cache._seen_tokens = k.shape[2]
            self.per_sequence_caches[seq.sequence_id] = cache

    def split_to_sequences(self, batch_cache: DynamicCache, sequences: list) -> list:
        """Split batch cache and return list of per-sequence caches (also stores them)."""
        self.disassemble_batch_cache(batch_cache, sequences)
        return [self.per_sequence_caches[seq.sequence_id] for seq in sequences]
