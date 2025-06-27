from __future__ import annotations

"""Simple wrapper to use LMCache with HuggingFace models."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.v1.cache_engine import LMCacheEngineBuilder


@dataclass
class LMCacheWrapper:
    """Helper class around :class:`LMCacheEngine` for storing and retrieving KV caches."""

    engine_id: str
    engine: object

    @staticmethod
    def create(model, *, chunk_size: int = 256, backend: str = "cuda", engine_id: str = "default") -> "LMCacheWrapper":
        """Create a :class:`LMCacheWrapper` for the given model."""
        cfg = model.config
        head_dim = cfg.hidden_size // getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        kv_shape = (cfg.num_hidden_layers, 2, chunk_size, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), head_dim)
        config = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend=backend)
        metadata = LMCacheEngineMetadata(
            model_name=getattr(cfg, "_name_or_path", "model"),
            world_size=1,
            worker_id=0,
            fmt="huggingface",
            kv_dtype=getattr(model, "dtype", torch.float16),
            kv_shape=kv_shape,
        )
        engine = LMCacheEngineBuilder.get_or_create(engine_id, config, metadata)
        return LMCacheWrapper(engine_id=engine_id, engine=engine)

    def store(self, tokens: torch.Tensor, kv_cache: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store KV caches for *tokens* into the cache engine."""
        self.engine.store(tokens, kv_cache)

    def retrieve(self, tokens: torch.Tensor):
        """Retrieve cached KV pairs for *tokens*.

        Returns the KV cache tuple or an empty tuple if nothing was found.
        """
        kv_cache, _ = self.engine.retrieve(tokens, return_tuple=True)
        return kv_cache
