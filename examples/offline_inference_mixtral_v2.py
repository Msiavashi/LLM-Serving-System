"""Example: Mixtral 8x7B with Priority Scheduler using the new QLLM v2 architecture.

Uses ModelAdapter (no model subclassing) + QllmEngine + SamplingParams.
The ModelAdapter auto-detects MoE layers and injects queue-aware wrappers.
"""
import sys
import os
import time
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory
from utils import generate_prompts


def main():
    np.random.seed(42)

    # Load Mixtral via ModelAdapter (no subclassing!)
    print("Loading Mixtral 8x7B via ModelAdapter...")
    adapter = ModelAdapter.from_name("mixtral", rank=0)
    print(f"  MoE detected: {adapter.is_moe}")
    print(f"  MoE blocks: {len(adapter._moe_block_names)}")

    # Inject queue-aware wrappers into MoE blocks
    adapter.inject_queues()
    print(f"  Queue wrappers injected: {len(adapter.moe_wrappers)}")

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

    # Create engine (without queue preemption for now — standard MoE routing)
    cache_manager = SequenceCacheManager(cache_type="dynamic")
    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=cache_manager,
        sampling_params=sampling_params,
        use_queues=False,  # Standard MoE (no per-expert queuing)
    )

    # Create priority scheduler
    scheduler = SchedulerFactory.create_scheduler(
        name="priority",
        engine=engine,
        tokenizer=adapter.tokenizer,
        batch_size=16,
    )

    # Set scheduler on adapter (for future preemption support)
    adapter.set_scheduler(scheduler)

    # Generate prompts
    prompts = generate_prompts(64, 16, adapter.tokenizer)
    num_ls = int(len(prompts) * 0.2)
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    high_priority_indices = np.argsort(exp_scores)[-num_ls:]

    for i, prompt in enumerate(prompts):
        priority = 1 if i in high_priority_indices else 0
        scheduler.add_sequence_to_queue(prompt, priority=priority)

    print(f"\nProcessing {len(prompts)} requests ({num_ls} LS, {len(prompts)-num_ls} BE)...")
    start = time.time()
    results = scheduler.run_scheduler()
    elapsed = time.time() - start

    # Report
    token_counts = [seq.generated_tokens.numel() for seq in results]
    total_tokens = sum(token_counts)
    ls_results = [seq for seq in results if seq.priority == 1]
    be_results = [seq for seq in results if seq.priority == 0]

    print(f"\nCompleted {len(results)} requests in {elapsed:.2f}s")
    print(f"Total tokens: {total_tokens}, Throughput: {total_tokens/elapsed:.2f} tok/s")
    print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}")

    if ls_results:
        ls_ta = [seq.finish_time - seq.arrival_time for seq in ls_results
                 if hasattr(seq, 'finish_time') and seq.finish_time]
        if ls_ta:
            print(f"LS turnaround: avg={np.mean(ls_ta):.2f}s")
    if be_results:
        be_ta = [seq.finish_time - seq.arrival_time for seq in be_results
                 if hasattr(seq, 'finish_time') and seq.finish_time]
        if be_ta:
            print(f"BE turnaround: avg={np.mean(be_ta):.2f}s")

    # Show sample outputs
    print("\nSample outputs:")
    for seq in results[:3]:
        text = seq.get_generated_text(adapter.tokenizer)
        print(f"  [{seq.sequence_id}] ({seq.generated_tokens.numel()} tok): {text[:80]}...")


if __name__ == "__main__":
    main()
