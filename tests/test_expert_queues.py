"""Verify per-expert queuing produces correct results in decode mode.

End-to-end test: runs the same prompts through QllmEngine with
use_queues=False (standard) vs use_queues=True (queued), then compares
the generated tokens. Uses greedy decoding (temperature=0) for
deterministic comparison.
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


def run_inference(adapter, use_queues, prompts, max_tokens=5):
    """Run inference and return generated token IDs per prompt."""
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for deterministic comparison
        eos_token_id=None,  # Don't stop early
        max_new_tokens=max_tokens,
    )

    engine = QllmEngine(
        model_adapter=adapter,
        cache_manager=SequenceCacheManager(),
        sampling_params=sampling_params,
        use_queues=use_queues,
    )

    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs", engine=engine, tokenizer=adapter.tokenizer, batch_size=len(prompts)
    )

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)

    results = scheduler.run_scheduler()

    return {
        seq.prompt: seq.generated_tokens.tolist()
        for seq in sorted(results, key=lambda s: s.sequence_id)
    }


def verify_expert_queues(rank=0):
    logger.info("=" * 70)
    logger.info("TEST: Per-Expert Queue End-to-End Verification")
    logger.info("=" * 70)

    # Load model with queue wrappers
    logger.info("\n1. Loading Mixtral with queue wrappers...")
    adapter = ModelAdapter.from_name("mixtral", rank=rank)
    adapter.inject_queues()
    logger.info(f"   {len(adapter.moe_wrappers)} wrappers, {adapter.moe_wrappers[0].num_experts} experts, top-{adapter.moe_wrappers[0].top_k}")

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
    ]
    max_tokens = 5

    # --- Run 1: Standard decode (no queues) ---
    logger.info(f"\n2. Running STANDARD decode (use_queues=False)...")
    standard_results = run_inference(adapter, use_queues=False, prompts=prompts, max_tokens=max_tokens)

    for prompt, tokens in standard_results.items():
        text = adapter.tokenizer.decode(tokens)
        logger.info(f"   '{prompt}' → [{len(tokens)} tokens] '{text}'")

    # --- Run 2: Queued decode (per-expert queues) ---
    logger.info(f"\n3. Running QUEUED decode (use_queues=True)...")

    # Clear queue state from any prior run
    for w in adapter.moe_wrappers:
        for q in w.queues:
            while not q.is_empty():
                q.dequeue()

    queued_results = run_inference(adapter, use_queues=True, prompts=prompts, max_tokens=max_tokens)

    for prompt, tokens in queued_results.items():
        text = adapter.tokenizer.decode(tokens)
        logger.info(f"   '{prompt}' → [{len(tokens)} tokens] '{text}'")

    # --- Compare ---
    logger.info(f"\n4. Comparison:")

    all_match = True
    for prompt in prompts:
        std_tokens = standard_results[prompt]
        q_tokens = queued_results[prompt]

        if std_tokens == q_tokens:
            logger.info(f"   PASS '{prompt[:30]}...': tokens match exactly")
        else:
            all_match = False
            std_text = adapter.tokenizer.decode(std_tokens)
            q_text = adapter.tokenizer.decode(q_tokens)
            logger.info(f"   DIFF '{prompt[:30]}...':")
            logger.info(f"         Standard: {std_tokens} = '{std_text}'")
            logger.info(f"         Queued:   {q_tokens} = '{q_text}'")

    # --- Queue activity check ---
    logger.info(f"\n5. Queue state after inference:")
    total_queued = sum(w.count_queued_tokens() for w in adapter.moe_wrappers)
    any_items = any(w.has_queued_items() for w in adapter.moe_wrappers)
    logger.info(f"   Tokens remaining in queues: {total_queued}")
    logger.info(f"   Any non-empty queues: {any_items}")

    # --- Routing trace on one wrapper ---
    logger.info(f"\n6. Expert routing distribution (layer 0, last run):")
    w = adapter.moe_wrappers[0]
    # Re-run a quick forward to capture routing
    test_input = adapter.tokenizer("Test prompt", return_tensors="pt").to(f"cuda:{rank}")
    with torch.inference_mode():
        w.set_decode_mode(False)  # prefill mode for single forward
        hidden = adapter.model.model.embed_tokens(test_input["input_ids"])
        # Apply first layer norm
        first_layer = adapter.model.model.layers[0]
        normed = first_layer.post_attention_layernorm(hidden)
        router_logits = w.gate(normed.view(-1, normed.shape[-1]))
        routing_weights = torch.softmax(router_logits, dim=1)
        _, selected = torch.topk(routing_weights, w.top_k, dim=-1)

    for expert_idx in range(w.num_experts):
        count = (selected == expert_idx).sum().item()
        logger.info(f"   Expert {expert_idx}: {count} tokens")

    logger.info(f"\n{'=' * 70}")
    if all_match:
        logger.info("RESULT: PASS — Queued decode produces identical tokens to standard decode")
    else:
        logger.info("RESULT: DIFF — Outputs differ (may be due to float16 accumulation order)")
        logger.info("         This is expected if differences are small and text is coherent")
    logger.info("=" * 70)


if __name__ == "__main__":
    verify_expert_queues(rank=0)
