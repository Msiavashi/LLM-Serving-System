"""Visualize per-expert queue fill patterns across layers and iterations.

Generates heatmaps showing how tokens accumulate in expert queues
at each layer, and how different thresholds affect when tokens proceed.
"""
import sys
import os
import json
import time
import copy
import logging

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.WARNING)

from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory


class InstrumentedQllmEngine(QllmEngine):
    """QllmEngine with queue state snapshots for visualization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = []  # List of {iteration, layer_idx, queue_lengths[expert]}
        self._iteration = 0

    def _run_queued_decode(self, batch):
        """Override to capture queue state at each layer."""
        self._iteration += 1
        sequences = batch.sequences
        base = self.model_adapter.model.model
        device = next(base.parameters()).device

        self.model_adapter.set_decode_mode(True)

        per_seq_caches = {}
        for seq in sequences:
            per_seq_caches[seq.sequence_id] = self.cache_manager.get_or_create(seq.sequence_id)

        input_ids = self._get_decode_tokens(sequences, device)
        all_hidden = base.embed_tokens(input_ids)

        active_seqs = list(sequences)
        active_hidden = all_hidden

        with torch.inference_mode():
            for layer_idx, layer in enumerate(base.layers):
                if not active_seqs:
                    # Capture empty state for remaining layers
                    wrapper = self._wrapper_map.get(layer_idx)
                    if wrapper:
                        ql = [wrapper.queues[e].size() for e in range(wrapper.num_experts)]
                        self.snapshots.append({
                            'iteration': self._iteration,
                            'layer': layer_idx,
                            'queue_lengths': ql,
                            'active_tokens': 0,
                            'completed_tokens': 0,
                        })
                    continue

                # --- Attention per-sequence ---
                attn_outputs = []
                for j, seq in enumerate(active_seqs):
                    h = active_hidden[j:j+1]
                    cache = per_seq_caches[seq.sequence_id]
                    past_len = cache.get_seq_length() if cache.key_cache else 0
                    cache_pos = torch.tensor([past_len], device=device, dtype=torch.long)
                    pos_ids = cache_pos.unsqueeze(0)
                    pos_emb = base.rotary_emb(h, pos_ids)

                    residual = h
                    normed = layer.input_layernorm(h)
                    attn_out, _ = layer.self_attn(
                        hidden_states=normed, position_embeddings=pos_emb,
                        attention_mask=None, past_key_value=cache,
                        use_cache=True, cache_position=cache_pos,
                    )
                    attn_outputs.append(residual + attn_out)

                active_hidden = torch.cat(attn_outputs, dim=0)

                # --- MoE with threshold ---
                wrapper = self._wrapper_map.get(layer_idx)
                if wrapper is not None:
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)

                    for j, seq in enumerate(active_seqs):
                        seq._deferred_residual = residual[j:j+1]

                    completed_h, completed_seqs = self._moe_with_threshold(
                        wrapper, normed, active_seqs
                    )

                    # Capture queue state AFTER routing and threshold processing
                    ql = [wrapper.queues[e].size() for e in range(wrapper.num_experts)]
                    self.snapshots.append({
                        'iteration': self._iteration,
                        'layer': layer_idx,
                        'queue_lengths': ql,
                        'active_tokens': len(active_seqs),
                        'completed_tokens': len(completed_seqs),
                    })

                    if completed_seqs:
                        active_hidden = torch.cat(completed_h, dim=0)
                        active_seqs = completed_seqs
                    else:
                        active_hidden = torch.zeros(0, 1, normed.shape[-1], device=device, dtype=normed.dtype)
                        active_seqs = []
                else:
                    residual = active_hidden
                    normed = layer.post_attention_layernorm(active_hidden)
                    ffn = getattr(layer, 'block_sparse_moe', getattr(layer, 'mlp', None))
                    if ffn:
                        out = ffn(normed)
                        if isinstance(out, tuple):
                            out = out[0]
                        active_hidden = residual + out

        # Final norm + lm_head
        if active_seqs:
            active_hidden = base.norm(active_hidden)
            logits = self.model_adapter.model.lm_head(active_hidden)
            for seq in active_seqs:
                self.cache_manager.per_sequence_caches[seq.sequence_id] = per_seq_caches[seq.sequence_id]
            kv_list = [per_seq_caches[seq.sequence_id] for seq in active_seqs]
            self.sampling_processor.sample_batch(logits, active_seqs, kv_list)
        else:
            for seq in sequences:
                self.cache_manager.per_sequence_caches[seq.sequence_id] = per_seq_caches[seq.sequence_id]

        return batch


def run_experiment(adapter, threshold, num_prompts=16, max_decode_iters=5):
    """Run decode with given threshold, capture queue snapshots."""
    # Clear queues
    for w in adapter.moe_wrappers:
        for q in w.queues:
            while not q.is_empty():
                q.dequeue()

    sp = SamplingParams(temperature=0.0, eos_token_id=None)
    engine = InstrumentedQllmEngine(
        model_adapter=adapter, cache_manager=SequenceCacheManager(),
        sampling_params=sp, use_queues=True, queue_threshold=threshold,
    )
    scheduler = SchedulerFactory.create_scheduler(
        'fcfs', engine=engine, tokenizer=adapter.tokenizer, batch_size=num_prompts,
    )

    # Use real prompts for interesting routing
    prompts = [
        "The capital of France is", "Water boils at a temperature of",
        "The theory of relativity was developed by", "Machine learning is",
        "The largest planet in our solar system is", "Photosynthesis converts",
        "The speed of light is approximately", "DNA stands for",
        "Quantum computing uses", "The human brain contains",
        "Artificial intelligence can", "The periodic table organizes",
        "Black holes are formed when", "The internet was invented",
        "Climate change is caused by", "Neural networks learn by",
    ][:num_prompts]

    for p in prompts:
        scheduler.add_sequence_to_queue(p)

    # Run scheduler but limit decode iterations
    from src.batching.batch import Batch
    from src.sequence.stage import Stage

    # Prefill first
    batch = scheduler.batch_policy.get_next_batch(scheduler.prefill_queue)
    if batch and batch.size() > 0:
        engine._run_standard(batch)  # Prefill always standard
        for seq in batch.sequences:
            seq.stage = Stage.DECODE
            scheduler.decode_queue.enqueue(seq)

    # Decode iterations with instrumentation
    for decode_iter in range(max_decode_iters):
        if scheduler.decode_queue.is_empty():
            break
        batch = scheduler.batch_policy.get_next_batch(scheduler.decode_queue)
        if batch is None or batch.size() == 0:
            break
        engine.run_batch(batch)
        # Re-enqueue unfinished sequences
        for seq in batch.sequences:
            if not seq.is_finished():
                scheduler.decode_queue.enqueue(seq)

    return engine.snapshots


def plot_heatmaps(all_snapshots, thresholds, output_dir):
    """Generate heatmap figures for each threshold."""
    os.makedirs(output_dir, exist_ok=True)

    for threshold, snapshots in zip(thresholds, all_snapshots):
        if not snapshots:
            continue

        num_experts = len(snapshots[0]['queue_lengths'])
        iterations = sorted(set(s['iteration'] for s in snapshots))
        num_layers = max(s['layer'] for s in snapshots) + 1

        # Create one heatmap per iteration
        fig, axes = plt.subplots(1, len(iterations), figsize=(4 * len(iterations), 6),
                                  squeeze=False)
        fig.suptitle(f'Expert Queue Lengths — Threshold = {threshold}',
                     fontsize=14, fontweight='bold')

        max_q = max(max(s['queue_lengths']) for s in snapshots) if snapshots else 1
        max_q = max(max_q, 1)

        for col, iteration in enumerate(iterations):
            ax = axes[0][col]
            # Build matrix: rows=experts, cols=layers
            matrix = np.zeros((num_experts, num_layers))
            completed_at_layer = np.zeros(num_layers)
            active_at_layer = np.zeros(num_layers)

            for s in snapshots:
                if s['iteration'] == iteration:
                    for e, ql in enumerate(s['queue_lengths']):
                        matrix[e, s['layer']] = ql
                    completed_at_layer[s['layer']] = s['completed_tokens']
                    active_at_layer[s['layer']] = s['active_tokens']

            im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max_q,
                          interpolation='nearest')

            # Annotate cells with values
            for e in range(num_experts):
                for l in range(num_layers):
                    val = int(matrix[e, l])
                    if val > 0:
                        ax.text(l, e, str(val), ha='center', va='center',
                               fontsize=6, color='black' if val < max_q * 0.7 else 'white')

            ax.set_xlabel('Layer ID', fontsize=10)
            if col == 0:
                ax.set_ylabel('Expert ID', fontsize=10)
            ax.set_title(f'Decode Iter {iteration}', fontsize=10)
            ax.set_xticks(range(0, num_layers, 4))
            ax.set_yticks(range(num_experts))

        plt.colorbar(im, ax=axes[0], label='Queue Length', shrink=0.8)
        plt.tight_layout()

        path = os.path.join(output_dir, f'queue_heatmap_threshold_{threshold}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # Summary comparison plot
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5),
                              squeeze=False)
    fig.suptitle('Expert Queue State at First Decode Iteration (All Thresholds)',
                 fontsize=14, fontweight='bold')

    for col, (threshold, snapshots) in enumerate(zip(thresholds, all_snapshots)):
        ax = axes[0][col]
        if not snapshots:
            ax.set_title(f'T={threshold}\n(no data)')
            continue

        num_experts = len(snapshots[0]['queue_lengths'])
        num_layers = max(s['layer'] for s in snapshots) + 1
        matrix = np.zeros((num_experts, num_layers))

        for s in snapshots:
            if s['iteration'] == 1:
                for e, ql in enumerate(s['queue_lengths']):
                    matrix[e, s['layer']] = ql

        max_q = max(matrix.max(), 1)
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max_q,
                      interpolation='nearest')

        for e in range(num_experts):
            for l in range(num_layers):
                val = int(matrix[e, l])
                if val > 0:
                    ax.text(l, e, str(val), ha='center', va='center',
                           fontsize=5, color='black' if val < max_q * 0.7 else 'white')

        ax.set_xlabel('Layer ID')
        if col == 0:
            ax.set_ylabel('Expert ID')

        # Count how many layers had completed tokens
        layers_completed = sum(1 for s in snapshots
                               if s['iteration'] == 1 and s['completed_tokens'] > 0)
        ax.set_title(f'Threshold={threshold}\n({layers_completed} layers passed)')

        ax.set_xticks(range(0, num_layers, 4))
        ax.set_yticks(range(num_experts))

    plt.tight_layout()
    path = os.path.join(output_dir, 'queue_comparison_all_thresholds.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("Loading Mixtral 8x7B...")
    adapter = ModelAdapter.from_name("mixtral", rank=0)
    adapter.inject_queues()
    print(f"  {len(adapter.moe_wrappers)} MoE layers, "
          f"{adapter.moe_wrappers[0].num_experts} experts, "
          f"top-{adapter.moe_wrappers[0].top_k}")

    thresholds = [1, 2, 4, 8, 16]
    all_snapshots = []
    output_dir = "results/queue_heatmaps"

    for threshold in thresholds:
        print(f"\n--- Experiment: threshold={threshold} ---")
        t0 = time.time()
        snapshots = run_experiment(adapter, threshold, num_prompts=16, max_decode_iters=5)
        elapsed = time.time() - t0
        all_snapshots.append(snapshots)

        # Summary
        iters_with_output = len(set(
            s['iteration'] for s in snapshots if s['completed_tokens'] > 0
        ))
        total_completed = sum(s['completed_tokens'] for s in snapshots)
        max_queue = max((max(s['queue_lengths']) for s in snapshots), default=0)
        print(f"  Time: {elapsed:.1f}s, Snapshots: {len(snapshots)}, "
              f"Iters with output: {iters_with_output}, "
              f"Total completed: {total_completed}, Max queue: {max_queue}")

    print(f"\nGenerating heatmaps...")
    plot_heatmaps(all_snapshots, thresholds, output_dir)

    # Save raw data
    data_path = os.path.join(output_dir, 'queue_data.json')
    with open(data_path, 'w') as f:
        json.dump({
            'thresholds': thresholds,
            'experiments': [
                {'threshold': t, 'snapshots': snaps}
                for t, snaps in zip(thresholds, all_snapshots)
            ]
        }, f, indent=2)
    print(f"  Raw data: {data_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
