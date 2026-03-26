"""Example: Llama 3 8B inference using the new QLLM v2 architecture.

Uses ModelAdapter + QllmEngine + SamplingParams instead of model subclassing.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory


def main():
    # Load model via ModelAdapter (no subclassing)
    print("Loading Llama 3 8B via ModelAdapter...")
    adapter = ModelAdapter.from_name("llama3_8b", rank=0)
    print(f"  Model loaded. MoE: {adapter.is_moe}")
    print(f"  Tokenizer: {adapter.tokenizer.__class__.__name__}")

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        frequency_penalty=0.2,
        eos_token_id=adapter.tokenizer.eos_token_id,
        exclude_token_ids=[
            t for t in [adapter.tokenizer.pad_token_id, adapter.tokenizer.unk_token_id]
            if t is not None
        ],
        max_new_tokens=20,
    )

    # Create engine
    cache_manager = SequenceCacheManager(cache_type="dynamic")
    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=cache_manager,
        sampling_params=sampling_params,
    )

    # Create scheduler (reusing existing FCFS scheduler)
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs",
        engine=engine,
        tokenizer=adapter.tokenizer,
        batch_size=4,
    )

    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "What are the main differences between classical and quantum computing?",
    ]

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)

    print(f"\nProcessing {len(prompts)} requests...")
    start = time.time()
    results = scheduler.run_scheduler()
    elapsed = time.time() - start

    print(f"\nCompleted {len(results)} requests in {elapsed:.2f}s\n")
    for seq in results:
        text = seq.get_generated_text(adapter.tokenizer)
        if adapter.tokenizer.pad_token:
            text = text.replace(adapter.tokenizer.pad_token, "").strip()
        print(f"[{seq.sequence_id}] Prompt: {seq.prompt}")
        print(f"    Output ({seq.generated_tokens.numel()} tokens): {text}")
        print()


if __name__ == "__main__":
    main()
