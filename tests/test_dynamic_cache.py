from src.sequence.sequence import Sequence
import unittest
from transformers.cache_utils import DynamicCache
from src.batching.batch import Batch
from src.models.model_factory import ModelFactory
from src.cache.unified_dynamic_cache import UnifiedDynamicCache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class TestDynamicCache(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 1
        self.seq_length = 5
        self.hidden_size = 8
        self.num_heads = 2
        self.head_dim = self.hidden_size // self.num_heads
        self.dynamic_cache = DynamicCache()
        self.unified_dynamic_cache = UnifiedDynamicCache([DynamicCache()])
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    
    def test_kv_update(self):
        # Create data for a single batch
        key_states = torch.randn(self.batch_size, self.num_heads, self.seq_length, self.head_dim)
        value_states = torch.randn(self.batch_size, self.num_heads, self.seq_length, self.head_dim)
        
        # First update - this should initialize the caches
        dynamic_cache_keys, dynamic_cache_values = self.dynamic_cache.update(key_states, value_states, layer_idx=0)
        
        # Now update the unified cache with the same inputs
        updated_keys, updated_values = self.unified_dynamic_cache.update(key_states, value_states, layer_idx=0)

        self.assertEqual(dynamic_cache_keys.shape, updated_keys.shape)
        
        self.assertEqual(dynamic_cache_values.shape, updated_values.shape)
        
        # Check if the updated keys and values are the same
        self.assertTrue(torch.equal(updated_keys, dynamic_cache_keys))
        self.assertTrue(torch.equal(updated_values, dynamic_cache_values))
        
    def test_updating_layer_i(self):
        # Create data for a single batch
        key_states = torch.randn(self.batch_size, self.num_heads, self.seq_length, self.head_dim)
        value_states = torch.randn(self.batch_size, self.num_heads, self.seq_length, self.head_dim)
        
        # First update - this should initialize the caches
        dynamic_cache_keys, dynamic_cache_values = self.dynamic_cache.update(key_states, value_states, layer_idx=1)
        
        # Now update the unified cache with the same inputs
        updated_keys, updated_values = self.unified_dynamic_cache.update(key_states, value_states, layer_idx=1)

        # Check if the updated keys and values are the same
        self.assertTrue(torch.equal(updated_keys, dynamic_cache_keys))
        self.assertTrue(torch.equal(updated_values, dynamic_cache_values))
        
        
    def test_output_cache(self):
        # Part 1: Get cache from HuggingFace model
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            self.skipTest(f"model unavailable: {e}")
        
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms."
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.device)
        
        with torch.no_grad():
            hf_outputs = hf_model(**inputs, use_cache=True)
        
        hf_cache = hf_outputs.past_key_values
        
        # Part 2: Get cache from our model
        model_instances, our_tokenizer = ModelFactory.create_model("llama3_8b", rank=0)
        
        sequences = []
        for prompt in prompts:
            sequence = Sequence(prompt, our_tokenizer, stage="PREFILL")
            sequences.append(sequence)
        
        our_model = model_instances[0].model
        batch = Batch(sequences=sequences)
        
        with torch.no_grad():
            batch, our_outputs = our_model(batch, use_cache=True)
        
        our_cache = our_outputs.past_key_values
        
        assert len(hf_cache) == len(our_cache), "Cache lengths do not match"
        
        # assert class type of each cache
        assert isinstance(hf_cache, DynamicCache)
        assert isinstance(our_cache, UnifiedDynamicCache)
        # Assert the get seq length methods to return same thing
        assert hf_cache.get_seq_length() == our_cache.get_seq_length()
        # Assert the get usable length methods to return same thing
        assert hf_cache.get_usable_length(100, 12) == our_cache.get_usable_length(100, 12)
        # Assert the update method to return same thing use random data with past_key_values shape
        random_key_states = torch.randn(hf_cache[0][0].shape, device="cuda:0")  # First layer, key states
        random_value_states = torch.randn(hf_cache[0][1].shape, device="cuda:0")  # First layer, value states
        hf_updated_keys, hf_updated_values = hf_cache.update(random_key_states, random_value_states, layer_idx=0)
        our_updated_keys, our_updated_values = our_cache.update(random_key_states, random_value_states, layer_idx=0)
        assert hf_updated_keys.shape == our_updated_keys.shape, "Updated keys shape do not match"
        assert hf_updated_values.shape == our_updated_values.shape, "Updated values shape do not match"
        
        
          
if __name__ == "__main__":
    unittest.main()