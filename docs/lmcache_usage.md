# Using LMCache

LMCache can be used to persist KV caches between requests to reduce the cost of
the prefill stage. Support for LMCache is integrated into the unified dynamic
cache used by the schedulers. When enabled, KV caches are automatically stored
and retrieved from an LMCache server.

## Installation

Install the extra dependencies:

```bash
pip install lmcache py-cpuinfo
```

## Basic example

The example `examples/offline_llama_8b_lmcache.py` shows how to use LMCache with
the provided Llama 8B model:

```python
model_instances, tokenizer = ModelFactory.create_model(
    "llama3_8b", rank=0, use_lmcache=True
)
```

When `use_lmcache` is enabled, KV caches will be transparently persisted using
LMCache. See `examples/offline_llama_8b_lmcache.py` for a complete example.

Access to the Llama checkpoints might require authentication from
Hugging Face.
