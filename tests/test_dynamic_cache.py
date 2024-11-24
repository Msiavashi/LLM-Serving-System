import unittest
import torch
import time
from src.cache.dynamic_cache import DynamicCacheEx

class TestDynamicCacheEx(unittest.TestCase):
    def setUp(self):
        # Initialize test data with different sizes
        self.batch_sizes = [2, 4, 8, 16, 32]
        self.seq_lengths = [10, 100, 1000]
        self.num_heads = 8
        self.head_dim = 128
        
    def create_test_cache(self, batch_size, seq_length):
        cache = DynamicCacheEx()
        # Create sample key and value tensors
        key = torch.randn(batch_size, self.num_heads, seq_length, self.head_dim)
        value = torch.randn(batch_size, self.num_heads, seq_length, self.head_dim)
        for layer_idx in range(32):  # 32 layers
            cache.update(key, value, layer_idx=layer_idx)
        return cache

    def test_split_kv_cache_latency(self):
        print("\nTesting split_kv_cache latency:")
        latencies = []
        
        for batch_size in self.batch_sizes:
            for seq_length in self.seq_lengths:
                cache = self.create_test_cache(batch_size, seq_length)
                
                # Measure split operation time
                start_time = time.perf_counter()
                split_caches = cache.split_kv_cache(batch_size)
                latency = (time.perf_counter() - start_time) * 1000
                
                latencies.append({
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'latency_ms': latency
                })
                
                # Verify split operation
                self.assertEqual(len(split_caches), batch_size)
                for split_cache in split_caches:
                    self.assertEqual(split_cache.key_cache[0].shape[0], 1)
                    self.assertEqual(split_cache.value_cache[0].shape[0], 1)
                
                print(f"Batch size: {batch_size}, Seq length: {seq_length}, "
                      f"Latency: {latency:.2f}ms")

    def test_merge_kv_cache_latency(self):
        print("\nTesting merge_kv_cache latency (no padding):")
        latencies = []
        
        for batch_size in self.batch_sizes:
            for seq_length in self.seq_lengths:
                # Create and split cache first
                cache = self.create_test_cache(batch_size, seq_length)
                split_caches = cache.split_kv_cache(batch_size)
                
                # Measure merge operation time
                start_time = time.perf_counter()
                merged_cache = DynamicCacheEx.merge_kv_caches(split_caches)
                latency = (time.perf_counter() - start_time) * 1000
                
                latencies.append({
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'latency_ms': latency
                })
                
                # Verify merge operation
                self.assertEqual(merged_cache.key_cache[0].shape[0], batch_size)
                self.assertEqual(merged_cache.value_cache[0].shape[0], batch_size)
                
                print(f"Batch size: {batch_size}, Seq length: {seq_length}, "
                      f"Latency: {latency:.2f}ms")

    def test_padding_correctness(self):
        # Test padding with different sequence lengths
        seq_lengths = [50, 100]  # Create tensors with different lengths
        caches = []
        
        for seq_len in seq_lengths:
            cache = DynamicCacheEx()
            key = torch.randn(1, self.num_heads, seq_len, self.head_dim)
            value = torch.randn(1, self.num_heads, seq_len, self.head_dim)
            for layer_idx in range(32):  # 32 layers
                cache.update(key, value, layer_idx=layer_idx)
            caches.append(cache)
        
        # Measure merge operation time    
        start_time = time.perf_counter()
        merged_cache = DynamicCacheEx.merge_kv_caches(caches)
        latency = (time.perf_counter() - start_time) * 1000
        
        print(f"Padding merge latency: {latency:.2f}ms")
        
        # Verify padding
        self.assertEqual(merged_cache.key_cache[0].shape[2], 100)
        self.assertEqual(merged_cache.value_cache[0].shape[2], 100)

if __name__ == '__main__':
    unittest.main()