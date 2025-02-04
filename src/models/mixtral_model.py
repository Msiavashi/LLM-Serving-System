import torch
from transformers import MixtralForCausalLM
from src.batching.batch import Batch
# from src.cache.dynamic_cache import DynamicCacheEx as DynamicCache
from src.cache.unified_dynamic_cache import UnifiedDynamicCache as DynamicCache

class MyCustomMixtral(MixtralForCausalLM):
    def forward(self, batch: Batch, **kwargs):
        # Set running sequences
        self.running_sequences = batch.sequences
        
        # Extract model inputs
        input_ids, attention_mask, past_key_values = self._prepare_inputs(batch)
        
        # Forward pass
        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        
        # Process outputs
        return self._process_outputs(outputs, len(self.running_sequences))
    
    def _prepare_inputs(self, batch: Batch):
        input_ids_list, attention_mask_list, past_key_values_list = batch.model_inputs.get_all_inputs()
        
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        if past_key_values_list:
            # past_key_values = DynamicCache.merge_kv_caches(past_key_values_list)
            past_key_values = DynamicCache(past_key_values_list)
        else:
            past_key_values = DynamicCache()
        
        return input_ids, attention_mask, past_key_values

    def _process_outputs(self, outputs, num_sequences):
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        # Split kv_cache and create a new batch
        # split_kv_cache = kv_cache.split_kv_cache(num_sequences)
        split_kv_cache = kv_cache.split_kv_cache()
        new_batch = Batch(self.running_sequences)
        new_batch.update_sequences(logits, split_kv_cache)
        
        return new_batch
