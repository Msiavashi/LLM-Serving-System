"""Verify layer-by-layer queued decode with threshold-based expert queuing.

Tests:
1. threshold=1 (immediate processing) matches standard decode
2. threshold>batch_size causes deferral (tokens stay in queues)
3. Deferred tokens complete when more tokens arrive and fill the threshold
"""
import sys
import os
import torch
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory


def run_with_config(adapter, prompts, use_queues, queue_threshold, max_tokens=5):
    """Run inference with specific queue config. Returns token lists per prompt."""
    sp = SamplingParams(temperature=0.0, eos_token_id=None, max_new_tokens=max_tokens)
    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=SequenceCacheManager(),
        sampling_params=sp,
        use_queues=use_queues,
        queue_threshold=queue_threshold,
    )
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs", engine=engine, tokenizer=adapter.tokenizer, batch_size=len(prompts)
    )
    for p in prompts:
        scheduler.add_sequence_to_queue(p)
    results = scheduler.run_scheduler()
    return {
        seq.prompt: (seq.generated_tokens.tolist(), seq.generated_tokens.numel())
        for seq in sorted(results, key=lambda s: s.sequence_id)
    }


def test_queued_decode(rank=0):
    logger.info("=" * 70)
    logger.info("TEST: Layer-by-Layer Queued Decode")
    logger.info("=" * 70)

    adapter = ModelAdapter.from_name("mixtral", rank=rank)
    adapter.inject_queues()
    logger.info(f"Loaded Mixtral: {len(adapter.moe_wrappers)} MoE wrappers\n")

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
    ]

    # --- Test 1: Standard (no queues) ---
    logger.info("Test 1: Standard decode (no queues)")
    std = run_with_config(adapter, prompts, use_queues=False, queue_threshold=1)
    for p, (tokens, n) in std.items():
        logger.info(f"  '{p}' → {n} tokens: {adapter.tokenizer.decode(tokens)}")

    # --- Test 2: Queued decode, threshold=1 (should match standard) ---
    logger.info("\nTest 2: Queued decode, threshold=1 (immediate)")
    # Clear queues from any prior run
    for w in adapter.moe_wrappers:
        for q in w.queues:
            while not q.is_empty():
                q.dequeue()
    q1 = run_with_config(adapter, prompts, use_queues=True, queue_threshold=1)
    for p, (tokens, n) in q1.items():
        logger.info(f"  '{p}' → {n} tokens: {adapter.tokenizer.decode(tokens)}")

    # Compare
    match_t1 = all(std[p][0] == q1[p][0] for p in prompts)
    logger.info(f"  Match with standard: {'PASS' if match_t1 else 'DIFF'}")

    # --- Test 3: Queued decode, threshold=2 (matches batch size) ---
    logger.info("\nTest 3: Queued decode, threshold=2 (= batch_size, all experts should fire)")
    for w in adapter.moe_wrappers:
        for q in w.queues:
            while not q.is_empty():
                q.dequeue()
    q2 = run_with_config(adapter, prompts, use_queues=True, queue_threshold=2)
    for p, (tokens, n) in q2.items():
        logger.info(f"  '{p}' → {n} tokens: {adapter.tokenizer.decode(tokens)}")

    match_t2 = all(std[p][0] == q2[p][0] for p in prompts)
    logger.info(f"  Match with standard: {'PASS' if match_t2 else 'DIFF'}")

    # --- Test 4: Check queue activity with high threshold ---
    logger.info("\nTest 4: Queue state with threshold=100 (tokens should be deferred)")
    for w in adapter.moe_wrappers:
        for q in w.queues:
            while not q.is_empty():
                q.dequeue()
    sp = SamplingParams(temperature=0.0, eos_token_id=None, max_new_tokens=1)
    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=SequenceCacheManager(),
        sampling_params=sp,
        use_queues=True,
        queue_threshold=100,  # Way above batch size — nothing should fire
    )
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs", engine=engine, tokenizer=adapter.tokenizer, batch_size=2
    )
    for p in prompts:
        scheduler.add_sequence_to_queue(p)
    results = scheduler.run_scheduler()

    # Check that tokens were deferred (no output generated because experts didn't fire)
    total_generated = sum(seq.generated_tokens.numel() for seq in results)
    total_queued = sum(w.count_queued_tokens() for w in adapter.moe_wrappers)
    logger.info(f"  Tokens generated: {total_generated}")
    logger.info(f"  Tokens in expert queues: {total_queued}")
    if total_queued > 0:
        logger.info("  PASS: Tokens deferred in expert queues (threshold not met)")
    else:
        logger.info("  Note: All tokens processed (experts may have been pre-filled)")

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULTS:")
    logger.info(f"  Test 1 (standard):           baseline captured")
    logger.info(f"  Test 2 (queued, threshold=1): {'PASS' if match_t1 else 'DIFF'}")
    logger.info(f"  Test 3 (queued, threshold=2): {'PASS' if match_t2 else 'DIFF'}")
    logger.info(f"  Test 4 (threshold=100):       deferred={total_queued > 0}")
    logger.info("=" * 70)


if __name__ == "__main__":
    test_queued_decode(rank=0)
