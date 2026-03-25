"""
QLLM Baseline Benchmark Harness

Runs Llama 8B (FCFS) and Mixtral 8x7B (Priority) benchmarks,
capturing throughput, latency, memory, and sample outputs.
"""
import sys
import os
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory

# Ensure examples/utils.py is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))
from utils import generate_prompts


LLAMA_PROMPTS = [
    "Answer in one word. What is the last word in the sentence that you are reading right now?",
    "Write a short story about a robot learning to paint.",
    "Explain the theory of relativity in simple terms.",
    "List three benefits of regular exercise.",
    "Describe the process of photosynthesis in detail, including the role of chlorophyll and sunlight.",
    "Summarize the plot of 'Pride and Prejudice' in one sentence.",
    "What are the main differences between classical and quantum computing?",
    "How does the human brain process information?",
    "What are the key principles of effective time management?",
    "Explain the significance of the Turing test in artificial intelligence.",
    "Describe the impact of climate change on global ecosystems.",
    "Discuss the ethical implications of genetic engineering in humans, including potential benefits and risks.",
]


def benchmark_llama_8b(rank=0):
    """Benchmark Llama 3 8B with FCFS scheduler."""
    print("=" * 80)
    print("BENCHMARK: Llama 3 8B — FCFS Scheduler")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.max_memory_allocated() / 1024**2

    model_instances, tokenizer = ModelFactory.create_model("llama3_8b", rank=rank)
    model = model_instances[0].model

    mem_after_load = torch.cuda.max_memory_allocated() / 1024**2

    engine = EngineFactory.create_engine("model", model=model)
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs", engine=engine, tokenizer=tokenizer, batch_size=4
    )

    for prompt in LLAMA_PROMPTS:
        scheduler.add_sequence_to_queue(prompt)

    start_time = time.time()
    results = scheduler.run_scheduler()
    end_time = time.time()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    # Mark completion times
    for seq in results:
        if not hasattr(seq, 'finish_time') or seq.finish_time is None:
            seq.finish_time = end_time

    # Collect sample outputs
    sample_outputs = []
    for seq in results[:4]:
        generated_text = seq.get_generated_text(tokenizer)
        if tokenizer.pad_token:
            generated_text = generated_text.replace(tokenizer.pad_token, "").strip()
        sample_outputs.append({
            "prompt": seq.prompt,
            "generated": generated_text,
            "num_tokens": seq.generated_tokens.numel(),
        })

    total_tokens = sum(seq.generated_tokens.numel() for seq in results)
    total_time = end_time - start_time

    metrics = {
        "model": "llama3_8b",
        "scheduler": "fcfs",
        "batch_size": 4,
        "num_prompts": len(LLAMA_PROMPTS),
        "total_time_s": round(total_time, 2),
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": round(total_tokens / total_time, 2),
        "gpu_memory_model_load_mb": round(mem_after_load - mem_before, 1),
        "gpu_memory_peak_mb": round(mem_peak, 1),
        "sample_outputs": sample_outputs,
    }

    print(f"\nResults: {total_tokens} tokens in {total_time:.2f}s = {total_tokens/total_time:.2f} tok/s")
    print(f"GPU Memory: {mem_after_load - mem_before:.0f} MB (model) / {mem_peak:.0f} MB (peak)")

    # Cleanup
    del model, engine, scheduler, model_instances, results
    torch.cuda.empty_cache()

    return metrics


def benchmark_mixtral_priority(rank=0):
    """Benchmark Mixtral 8x7B with Priority scheduler + per-expert queues."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Mixtral 8x7B — Priority Scheduler (per-expert queues)")
    print("=" * 80)

    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.max_memory_allocated() / 1024**2

    model_instances, tokenizer = ModelFactory.create_mixtral_queue_model(rank=rank)
    model = model_instances[0].model

    mem_after_load = torch.cuda.max_memory_allocated() / 1024**2

    engine = EngineFactory.create_engine("model", model=model)
    scheduler = SchedulerFactory.create_scheduler(
        name="priority", engine=engine, tokenizer=tokenizer, batch_size=16
    )
    model.set_scheduler(scheduler)

    prompts = generate_prompts(256, 16, tokenizer)

    num_latency_sensitive = int(len(prompts) * 0.2)
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    high_priority_indices = np.argsort(exp_scores)[-num_latency_sensitive:]

    for i, prompt in enumerate(prompts):
        priority = 1 if i in high_priority_indices else 0
        scheduler.add_sequence_to_queue(prompt, priority=priority)

    start_time = time.time()
    results = scheduler.run_scheduler()
    end_time = time.time()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    total_tokens = sum(seq.generated_tokens.numel() for seq in results)
    total_time = end_time - start_time

    # Collect sample outputs
    sample_outputs = []
    for seq in results[:4]:
        generated_text = seq.get_generated_text(tokenizer)
        if tokenizer.pad_token:
            generated_text = generated_text.replace(tokenizer.pad_token, "").strip()
        sample_outputs.append({
            "prompt": seq.prompt,
            "generated": generated_text,
            "num_tokens": seq.generated_tokens.numel(),
        })

    # Compute LS vs BE turnaround
    ls_turnaround = []
    be_turnaround = []
    for seq in results:
        if hasattr(seq, 'finish_time') and seq.finish_time and hasattr(seq, 'arrival_time'):
            ta = seq.finish_time - seq.arrival_time
            if seq.priority == 1:
                ls_turnaround.append(ta)
            else:
                be_turnaround.append(ta)

    metrics = {
        "model": "mixtral_8x7b_queue",
        "scheduler": "priority",
        "batch_size": 16,
        "num_prompts": 256,
        "num_ls_prompts": num_latency_sensitive,
        "total_time_s": round(total_time, 2),
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": round(total_tokens / total_time, 2),
        "avg_ls_turnaround_s": round(np.mean(ls_turnaround), 2) if ls_turnaround else None,
        "avg_be_turnaround_s": round(np.mean(be_turnaround), 2) if be_turnaround else None,
        "gpu_memory_model_load_mb": round(mem_after_load - mem_before, 1),
        "gpu_memory_peak_mb": round(mem_peak, 1),
        "sample_outputs": sample_outputs,
    }

    print(f"\nResults: {total_tokens} tokens in {total_time:.2f}s = {total_tokens/total_time:.2f} tok/s")
    print(f"LS turnaround: {np.mean(ls_turnaround):.2f}s" if ls_turnaround else "No LS results")
    print(f"BE turnaround: {np.mean(be_turnaround):.2f}s" if be_turnaround else "No BE results")
    print(f"GPU Memory: {mem_after_load - mem_before:.0f} MB (model) / {mem_peak:.0f} MB (peak)")

    # Cleanup
    del model, engine, scheduler, model_instances, results
    torch.cuda.empty_cache()

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QLLM Benchmark")
    parser.add_argument("--model", choices=["llama", "mixtral", "all"], default="all")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--output", type=str, default="results/baseline.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": []}

    if args.model in ("llama", "all"):
        metrics = benchmark_llama_8b(rank=args.rank)
        results["benchmarks"].append(metrics)

    if args.model in ("mixtral", "all"):
        metrics = benchmark_mixtral_priority(rank=args.rank)
        results["benchmarks"].append(metrics)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
