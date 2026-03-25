"""QllmEngine: Unified engine using HF-native forward pass, cache, and sampling.

Replaces the old ModelEngine + model subclass approach. Uses:
- ModelAdapter for model loading and MoE injection
- SequenceCacheManager for per-sequence KV cache (HF-native)
- SamplingProcessor for vectorized token sampling
"""
import logging
import torch
from typing import Optional

from src.engines.base_engine import BaseEngine
from src.models.model_adapter import ModelAdapter
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_processor import SamplingProcessor
from src.samplers.sampling_params import SamplingParams
from src.sequence.stage import Stage

logger = logging.getLogger(__name__)


class QllmEngine(BaseEngine):
    """Engine that uses HF-native model forward with per-sequence cache management."""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        cache_manager: Optional[SequenceCacheManager] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.model_adapter = model_adapter
        self.model = model_adapter.model  # For compatibility with existing scheduler interface
        self.cache_manager = cache_manager or SequenceCacheManager()
        self.sampling_params = sampling_params or SamplingParams()
        self.sampling_processor = SamplingProcessor(self.sampling_params)

    def run_batch(self, batch) -> "Batch":
        """Execute one forward pass for a batch of sequences.

        This method:
        1. Sets MoE wrappers to prefill/decode mode
        2. Prepares inputs in standard HF format
        3. Assembles per-sequence caches into batch cache
        4. Runs HF model forward (with native attention, cache, etc.)
        5. Splits batch cache back to per-sequence
        6. Samples next tokens
        """
        sequences = batch.sequences
        is_decode = batch.is_decode()

        # Set MoE wrappers mode
        self.model_adapter.set_decode_mode(is_decode)

        # Prepare inputs
        input_ids, attention_mask = self._prepare_inputs(batch)

        # Assemble batch cache from per-sequence caches
        past_key_values = self.cache_manager.assemble_batch_cache(sequences)

        # Forward pass with torch.inference_mode
        with torch.inference_mode():
            outputs = self.model_adapter.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Split cache back to per-sequence
        kv_caches = self.cache_manager.split_to_sequences(
            outputs.past_key_values, sequences
        )

        # Sample next tokens
        self.sampling_processor.sample_batch(outputs.logits, sequences, kv_caches)

        return batch

    def _prepare_inputs(self, batch):
        """Prepare model inputs in standard HF format."""
        from torch.nn.utils.rnn import pad_sequence

        sequences = batch.sequences
        is_decode = batch.is_decode()

        if is_decode:
            # Decode: use only the last generated token per sequence
            input_ids_list = []
            for seq in sequences:
                if seq.generated_tokens.numel() > 0:
                    input_ids_list.append(seq.generated_tokens[-1:])
                else:
                    input_ids_list.append(seq.input_ids[-1:])
        else:
            # Prefill: use full input_ids
            input_ids_list = [seq.input_ids for seq in sequences]

        # Pad to same length
        padded_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)

        # Build attention mask
        if is_decode:
            # For decode with KV cache, attention mask should cover full sequence length
            max_cache_len = max(
                (seq.get_total_sequence_length() for seq in sequences), default=1
            )
            attention_mask = torch.ones(
                len(sequences), max_cache_len,
                dtype=torch.long, device=padded_ids.device
            )
        else:
            attention_mask = (padded_ids != 0).long()

        return padded_ids, attention_mask
