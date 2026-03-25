"""Tests for the new QLLM v2 architecture components."""
import pytest
import torch
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.samplers.sampling_params import SamplingParams
from src.samplers.sampling_processor import SamplingProcessor
from src.samplers.sampling_metadata import SamplingMetadata
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.sequence.sequence_base import SequenceBase
from src.sequence.stage import Stage
from transformers.cache_utils import DynamicCache


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 0.7
        assert p.top_k == 50
        assert p.top_p == 0.9
        assert p.frequency_penalty == 0.2
        assert p.eos_token_id is None
        assert p.exclude_token_ids == []
        assert p.max_new_tokens == 20

    def test_custom(self):
        p = SamplingParams(temperature=0.0, top_k=10, eos_token_id=2)
        assert p.temperature == 0.0
        assert p.top_k == 10
        assert p.eos_token_id == 2


class TestSamplingMetadata:
    def test_increment_and_finish(self):
        sm = SamplingMetadata(num_tokens=5)
        assert not sm.is_finished()
        for _ in range(5):
            sm.increment_token_count()
        assert sm.is_finished()

    def test_force_finish(self):
        sm = SamplingMetadata(num_tokens=100)
        assert not sm.is_finished()
        sm.force_finish()
        assert sm.is_finished()


class TestStageEnum:
    def test_values(self):
        assert Stage.PREFILL == "prefill"
        assert Stage.DECODE == "decode"
        assert Stage.PREFILL.value == "prefill"

    def test_string_comparison(self):
        # Backward compatible with old string-based Stage
        assert Stage.PREFILL == "prefill"
        assert "decode" == Stage.DECODE


class TestSequenceBase:
    def test_thread_safe_ids(self):
        """Sequence IDs should be unique and monotonically increasing."""
        t = torch.tensor([1, 2, 3])
        m = torch.tensor([1, 1, 1])
        s1 = SequenceBase("p1", t, m, device="cpu")
        s2 = SequenceBase("p2", t, m, device="cpu")
        assert s2.sequence_id > s1.sequence_id

    def test_preallocated_buffer_update(self):
        """Update should use pre-allocated buffer, not O(n²) concat."""
        t = torch.tensor([1, 2, 3])
        m = torch.tensor([1, 1, 1])
        seq = SequenceBase("test", t, m, device="cpu",
                           sampling_metadata=SamplingMetadata(num_tokens=10))

        # Generate 5 tokens
        dummy_cache = DynamicCache()
        for i in range(5):
            seq.update(torch.tensor([100 + i]), dummy_cache)

        assert seq.generated_tokens.numel() == 5
        assert seq._gen_count == 5
        assert seq.generated_tokens[0].item() == 100
        assert seq.generated_tokens[4].item() == 104

    def test_is_finished(self):
        t = torch.tensor([1])
        m = torch.tensor([1])
        seq = SequenceBase("test", t, m, device="cpu",
                           sampling_metadata=SamplingMetadata(num_tokens=3))
        dummy_cache = DynamicCache()
        for _ in range(3):
            seq.update(torch.tensor([42]), dummy_cache)
        assert seq.is_finished()

    def test_timing_fields(self):
        t = torch.tensor([1])
        m = torch.tensor([1])
        seq = SequenceBase("test", t, m, device="cpu")
        assert seq.arrival_time is not None
        assert seq.first_token_time is None
        dummy_cache = DynamicCache()
        seq.update(torch.tensor([42]), dummy_cache)
        assert seq.first_token_time is not None


class TestSequenceCacheManager:
    def test_create_and_get(self):
        mgr = SequenceCacheManager()
        cache = mgr.get_or_create(0)
        assert isinstance(cache, DynamicCache)
        # Same ID returns same cache
        cache2 = mgr.get_or_create(0)
        assert cache is cache2

    def test_remove(self):
        mgr = SequenceCacheManager()
        mgr.get_or_create(0)
        mgr.remove(0)
        assert 0 not in mgr.per_sequence_caches

    def test_assemble_returns_none_for_empty(self):
        mgr = SequenceCacheManager()
        seq = MagicMock()
        seq.sequence_id = 0
        result = mgr.assemble_batch_cache([seq])
        assert result is None

    def test_assemble_disassemble_roundtrip(self):
        """Verify cache roundtrip: store → assemble → disassemble → verify."""
        mgr = SequenceCacheManager()

        # Create 2 sequences with fake caches
        seq1 = MagicMock()
        seq1.sequence_id = 100
        seq2 = MagicMock()
        seq2.sequence_id = 101

        # Manually create per-sequence caches with 1 layer
        cache1 = DynamicCache()
        k1 = torch.randn(1, 4, 5, 8)  # [batch=1, heads=4, seq_len=5, head_dim=8]
        v1 = torch.randn(1, 4, 5, 8)
        cache1.key_cache.append(k1)
        cache1.value_cache.append(v1)
        cache1._seen_tokens = 5

        cache2 = DynamicCache()
        k2 = torch.randn(1, 4, 5, 8)
        v2 = torch.randn(1, 4, 5, 8)
        cache2.key_cache.append(k2)
        cache2.value_cache.append(v2)
        cache2._seen_tokens = 5

        mgr.per_sequence_caches[100] = cache1
        mgr.per_sequence_caches[101] = cache2

        # Assemble
        batch_cache = mgr.assemble_batch_cache([seq1, seq2])
        assert batch_cache is not None
        assert batch_cache.key_cache[0].shape == (2, 4, 5, 8)  # batch dim merged

        # Disassemble
        mgr.disassemble_batch_cache(batch_cache, [seq1, seq2])
        rc1 = mgr.per_sequence_caches[100]
        rc2 = mgr.per_sequence_caches[101]
        assert rc1.key_cache[0].shape == (1, 4, 5, 8)
        assert rc2.key_cache[0].shape == (1, 4, 5, 8)

        # Values should match
        assert torch.allclose(rc1.key_cache[0], k1)
        assert torch.allclose(rc2.key_cache[0], k2)


class TestSamplingProcessor:
    def test_greedy_sampling(self):
        """Temperature=0 should produce argmax (greedy) output."""
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        processor = SamplingProcessor(params)

        # Create fake logits: batch_size=2, seq_len=1, vocab_size=10
        logits = torch.zeros(2, 1, 10)
        logits[0, 0, 7] = 100.0  # seq 0 should pick token 7
        logits[1, 0, 3] = 100.0  # seq 1 should pick token 3

        # Create mock sequences
        seqs = []
        for i in range(2):
            seq = SequenceBase(
                f"test{i}",
                torch.tensor([1]),
                torch.tensor([1]),
                device="cpu",
                sampling_metadata=SamplingMetadata(num_tokens=5),
            )
            seqs.append(seq)

        dummy_caches = [DynamicCache(), DynamicCache()]
        processor.sample_batch(logits, seqs, dummy_caches)

        assert seqs[0].generated_tokens[-1].item() == 7
        assert seqs[1].generated_tokens[-1].item() == 3

    def test_eos_detection(self):
        """EOS token should trigger force_finish."""
        params = SamplingParams(temperature=0.0, eos_token_id=2, max_new_tokens=10)
        processor = SamplingProcessor(params)

        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 2] = 100.0  # Will pick EOS token

        seq = SequenceBase(
            "test", torch.tensor([1]), torch.tensor([1]),
            device="cpu", sampling_metadata=SamplingMetadata(num_tokens=10),
        )

        processor.sample_batch(logits, [seq], [DynamicCache()])
        assert seq.is_finished()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
