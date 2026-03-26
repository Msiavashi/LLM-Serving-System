"""Example: Mixtral inference with ShareGPT dataset using QLLM v2 architecture."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from utils import read_shared_gpt_dataset
from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory


def main():
    np.random.seed(42)

    adapter = ModelAdapter.from_name("mixtral", rank=0)

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
        name="fcfs", engine=engine, tokenizer=adapter.tokenizer, batch_size=16
    )

    prompts = read_shared_gpt_dataset(
        "./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 32
    )

    prompt_lengths = [len(adapter.tokenizer.encode(p)) for p in prompts]
    avg_length = sum(prompt_lengths) / len(prompt_lengths)
    print(f"Number of prompts: {len(prompts)}, Average length: {avg_length:.0f} tokens")

    num_ls = int(len(prompts) * 0.2)
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    high_priority_indices = np.argsort(exp_scores)[-num_ls:]

    for i, prompt in enumerate(prompts):
        priority = 1 if i in high_priority_indices else 0
        scheduler.add_sequence_to_queue(prompt, priority=priority)

    results = scheduler.run_scheduler()

    for seq in results[:5]:
        text = seq.get_generated_text(adapter.tokenizer)
        print(f"Prompt: {seq.prompt[:60]}...\nGenerated: {text}\n")


if __name__ == "__main__":
    main()
