"""
This Mixin class is used for models implementing per-expert queueing.
For example see `src/models/mixtral_queue_model.py`
"""

from typing import List, Tuple, Optional

import torch
from src.batching.batch import Batch
from src.cache.unified_dynamic_cache import UnifiedDynamicCache as DynamicCache
from torch.nn.utils.rnn import pad_sequence


class ModelInputMixin:
    """
    A mixin class that provides utilities for preparing model inputs from batches.
    Handles padding, attention masking, and cache management.
    """
    
    def __init__(self):
        """Initialize the mixin with a empty running batch."""
        self.running_batch: Optional[Batch] = None
    
    def _pad_and_create_mask(self, input_ids_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad input tensors to the same length and create attention masks.
        
        Args:
            input_ids_list: List of input token tensors
        
        Returns:
            Tuple of (padded_input_ids, attention_mask)
        """
        # Normalize dimensions - ensure each tensor is 1D
        normalized_ids_list = [
            ids.squeeze(0) if ids.dim() == 2 and ids.size(0) == 1 else ids 
            for ids in input_ids_list
        ]
        
        # Pad sequences to max length in batch
        padded_inputs = pad_sequence(normalized_ids_list, batch_first=True, padding_value=0)
        
        # Create attention masks: 1 for real tokens, 0 for padding
        attention_mask = (padded_inputs != 0).long()
        
        return padded_inputs, attention_mask
    
    def _prepare_inputs(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[DynamicCache], Batch]:
        """
        Prepare model inputs from a batch.
        
        Args:
            batch: The batch of requests to process
        
        Returns:
            Tuple of (input_ids, attention_mask, past_key_values, running_batch)
        """
        self.running_batch = batch

        # Get inputs from batch, ignoring attention_mask_list as we'll create it from input_ids. Keeping it in Batch is still useful for preemption.
        input_ids_list, _, past_key_values_list = batch.model_inputs.get_all_inputs()

        # Process inputs and create masks
        input_ids, attention_mask = self._pad_and_create_mask(input_ids_list)
        
        # Create cache if available
        past_key_values = DynamicCache(past_key_values_list) if past_key_values_list else DynamicCache()
        
        return input_ids, attention_mask, past_key_values, self.running_batch