import torch
from typing import List, Any
from src.samplers.sampling_params import SamplingParams
from src.sequence.stage import Stage


class SamplingProcessor:
    """Vectorized token sampling with temperature, top-k, top-p, and EOS detection."""

    def __init__(self, params: SamplingParams):
        self.params = params

    def sample_batch(self, logits: torch.Tensor, sequences: list, kv_caches: List[Any]) -> None:
        """Sample next tokens for a batch of sequences.

        Args:
            logits: Model output logits, shape [batch_size, seq_len, vocab_size]
            sequences: List of Sequence objects
            kv_caches: List of per-sequence KV caches
        """
        batch_size = logits.shape[0]
        if batch_size != len(sequences):
            raise ValueError(
                f"Batch size mismatch: logits {batch_size} vs sequences {len(sequences)}"
            )

        # Extract last-position logits for the whole batch: [B, vocab_size]
        last_logits = logits[:, -1, :].clone()
        device = last_logits.device
        p = self.params

        # --- Frequency penalty (per-sequence) ---
        if p.frequency_penalty > 0:
            for i, seq in enumerate(sequences):
                if seq.generated_tokens.numel() > 0:
                    unique_tokens, counts = torch.unique(
                        seq.generated_tokens, return_counts=True
                    )
                    penalty = torch.zeros(
                        last_logits.shape[1], device=device, dtype=last_logits.dtype
                    )
                    penalty.index_add_(
                        0,
                        unique_tokens.to(device),
                        counts.to(last_logits.dtype) * p.frequency_penalty,
                    )
                    last_logits[i] -= penalty

        # --- Exclude tokens (batch-wide) ---
        if p.exclude_token_ids:
            exclude = torch.tensor(p.exclude_token_ids, device=device, dtype=torch.long)
            last_logits[:, exclude] = -float("inf")

        # --- Temperature + Sampling ---
        if p.temperature == 0:
            # Greedy: argmax for entire batch
            next_token_ids = torch.argmax(last_logits, dim=-1)  # [B]
        else:
            # Temperature scaling
            scaled_logits = last_logits / p.temperature

            # Batch top-k: [B, top_k]
            k = min(p.top_k, scaled_logits.shape[-1])
            top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)

            # Softmax over top-k candidates
            top_probs = torch.softmax(top_values, dim=-1)

            # Top-p (nucleus) filtering
            sorted_probs, sorted_idx = torch.sort(top_probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # Keep tokens where cumulative prob (before this token) < top_p
            nucleus_mask = (cumulative - sorted_probs) < p.top_p
            nucleus_mask[:, 0] = True  # always keep at least one token

            # Zero out probabilities outside nucleus and re-normalize
            filtered_probs = sorted_probs * nucleus_mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample from filtered distribution
            sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # [B]

            # Map back to vocabulary indices
            batch_range = torch.arange(batch_size, device=device)
            sampled_topk_idx = sorted_idx[batch_range, sampled_sorted_idx]
            next_token_ids = top_indices[batch_range, sampled_topk_idx]  # [B]

        # --- Update each sequence ---
        for i, seq in enumerate(sequences):
            token_id = next_token_ids[i].unsqueeze(0)
            seq.update(token_id, kv_caches[i])

            if seq.stage == Stage.PREFILL:
                seq.stage = Stage.DECODE

            # EOS detection
            if p.eos_token_id is not None and token_id.item() == p.eos_token_id:
                seq.sampling_metadata.force_finish()
