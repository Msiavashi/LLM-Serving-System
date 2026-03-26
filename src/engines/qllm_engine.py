"""QllmEngine: Uses standard HF model.forward() with per-expert queue wrappers.

The engine delegates ALL compute to HuggingFace's model.forward():
- Batched attention (SDPA / Flash Attention — native, efficient)
- KV cache management (DynamicCache — native)
- Quantization, torch.compile, etc. (pass-through)

Per-expert queuing is handled INSIDE the MoE wrapper's forward(), which is
called automatically by the model as part of its standard forward pass.
The wrapper routes tokens through per-expert FIFO queues, enabling:
- Expert-level batch size control (threshold parameter)
- Priority-aware preemption (scheduler feedback)
- Queue state monitoring

Priority scheduling happens at the SCHEDULER level (4-queue system from
Algorithm 1 in the paper). The engine processes whatever batch the scheduler
gives it. The wrapper checks for priority preemption during MoE processing.
"""
import logging
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from src.engines.base_engine import BaseEngine
from src.models.model_adapter import ModelAdapter
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_processor import SamplingProcessor
from src.samplers.sampling_params import SamplingParams
from src.sequence.stage import Stage

logger = logging.getLogger(__name__)


class QllmEngine(BaseEngine):
    """Engine using standard HF model.forward() with per-expert queue wrappers."""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        cache_manager: Optional[SequenceCacheManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        use_queues: bool = False,
        queue_threshold: int = 1,
    ):
        self.model_adapter = model_adapter
        self.model = model_adapter.model
        self.cache_manager = cache_manager or SequenceCacheManager()
        self.sampling_params = sampling_params or SamplingParams()
        self.sampling_processor = SamplingProcessor(self.sampling_params)
        self.use_queues = use_queues
        self.queue_threshold = queue_threshold

    def run_batch(self, batch) -> "Batch":
        """Execute one forward pass for a batch of sequences."""
        sequences = batch.sequences
        is_decode = batch.is_decode()

        # Configure MoE wrappers
        self.model_adapter.set_decode_mode(is_decode)
        for w in self.model_adapter.moe_wrappers:
            w.set_batch_context(
                sequences=sequences if is_decode and self.use_queues else None,
                use_queues=self.use_queues and is_decode,
            )

        # Prepare inputs
        input_ids, attention_mask = self._prepare_inputs(batch)

        # Assemble per-sequence KV caches into batch cache
        past_key_values = self.cache_manager.assemble_batch_cache(sequences)

        # Standard HF model.forward() — batched attention, native optimizations
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
        sequences = batch.sequences
        is_decode = batch.is_decode()

        if is_decode:
            ids = [
                seq.generated_tokens[-1:] if seq.generated_tokens.numel() > 0
                else seq.input_ids[-1:]
                for seq in sequences
            ]
        else:
            ids = [seq.input_ids for seq in sequences]

        padded = pad_sequence(ids, batch_first=True, padding_value=0)

        if is_decode:
            max_len = max(
                (seq.get_total_sequence_length() for seq in sequences), default=1
            )
            mask = torch.ones(
                len(sequences), max_len, dtype=torch.long, device=padded.device
            )
        else:
            mask = (padded != 0).long()

        return padded, mask
