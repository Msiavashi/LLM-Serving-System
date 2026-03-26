"""Example: Phi-3.5-MoE inference using QLLM v2 architecture."""
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
    adapter = ModelAdapter.from_name("phimoe", rank=0)
    print(f"Loaded {adapter.checkpoint} — MoE: {adapter.is_moe}")

    sampling_params = SamplingParams(
        temperature=0.7,
        eos_token_id=adapter.tokenizer.eos_token_id,
        exclude_token_ids=[
            t for t in [adapter.tokenizer.pad_token_id] if t is not None
        ],
    )

    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=SequenceCacheManager(),
        sampling_params=sampling_params,
    )

    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs", engine=engine, tokenizer=adapter.tokenizer, batch_size=4
    )

    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "Describe the process of photosynthesis in detail.",
        "What are the main differences between classical and quantum computing?",
        "How does the human brain process information?",
        "Explain the significance of the Turing test in artificial intelligence.",
        "Describe the impact of climate change on global ecosystems.",
    ]

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)

    print(f"Processing {len(prompts)} requests...")
    start = time.time()
    results = scheduler.run_scheduler()
    elapsed = time.time() - start

    print(f"\nCompleted {len(results)} requests in {elapsed:.2f}s\n")
    for seq in results:
        text = seq.get_generated_text(adapter.tokenizer)
        if adapter.tokenizer.pad_token:
            text = text.replace(adapter.tokenizer.pad_token, "").strip()
        print(f"Prompt: {seq.prompt}\nGenerated Text: {text}\n")


if __name__ == "__main__":
    main()
