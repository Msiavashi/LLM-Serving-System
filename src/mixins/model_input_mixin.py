"""
    This Mixin class in used for models implementing per-expert queueing. For example see `src/models/mixtral_queue_model.py`
"""

import torch
from src.batching.batch import Batch
from src.cache.unified_dynamic_cache import UnifiedDynamicCache as DynamicCache

class ModelInputMixin:
    
    def __init__(self):
        self.running_batch: Batch = None
    
    def _prepare_inputs(self, batch: Batch):
        
        self.running_batch = batch
        
        input_ids_list, attention_mask_list, past_key_values_list = batch.model_inputs.get_all_inputs()
        
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        past_key_values = DynamicCache(past_key_values_list) if past_key_values_list else None
        
        # # return empty tensors
        # input_ids = torch.zeros(1, 1, dtype=input_ids.dtype, device=input_ids.device)
        # attention_mask = torch.zeros(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        # past_key_values = DynamicCache([])
        # self.running_batch = Batch([])
        
        return input_ids, attention_mask, past_key_values, self.running_batch