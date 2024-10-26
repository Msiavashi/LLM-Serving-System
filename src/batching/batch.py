from typing import List, Union
from src.sequence import Sequence
import torch.nn.functional as F
from typing import Tuple
import torch

class ModelInputs:
    def __init__(self, input_ids, attention_masks, past_key_values):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.past_key_values = past_key_values
        
    def get_all_inputs(self):
        return self.input_ids, self.attention_masks, self.past_key_values
    
    def update(self, input_ids, attention_masks, past_key_values):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.past_key_values = past_key_values
        
    def restructure_kv_cache(self, past_key_values_list: List[Tuple[torch.Tensor, torch.Tensor]], num_layers: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Restructures the key-value cache for each layer.

        Args:
            past_key_values_list (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing the key-value cache for each sequence.
            num_layers (int): The number of layers.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor]]: A tuple containing the restructured key-value cache for each layer.
        """
        past_key_values = []
        for layer_idx in range(num_layers):
            key_states_list, value_states_list = [], []
            for seq_kv_cache in past_key_values_list:
                key_i, value_i = seq_kv_cache[layer_idx]
                key_states_list.append(key_i)
                value_states_list.append(value_i)
            key_states = torch.cat(key_states_list, dim=0)
            value_states = torch.cat(value_states_list, dim=0)
            past_key_values.append((key_states, value_states))
        return tuple(past_key_values)
        
    
class Batch:
    def __init__(self, sequences: List[Sequence] = None):
        self.sequences = sequences if sequences else []
        self._model_inputs = ModelInputs([], [], [])
    
    def add_sequence(self, sequence: Union[Sequence, List[Sequence]]):
        if isinstance(sequence, list):
            self.sequences.extend(sequence)
        else:
            self.sequences.append(sequence)

    def size(self):
        return len(self.sequences)
    
    def _preprocess_sequences(self):
        """
        Preprocesses the sequences by padding them to the maximum length and organizing them into input lists.

        Returns:
            None
        """
        input_ids_list, attention_mask_list, past_key_values_list = [], [], []
        max_length = max(seq.input_ids.size(0) for seq in self.sequences)
        
        for sequence in self.sequences:
            padding_length = max_length - sequence.input_ids.size(0)
            if sequence.stage == "prefill":
                if padding_length > 0:
                    sequence.input_ids = F.pad(sequence.input_ids, (0, padding_length), value=sequence.tokenizer.pad_token_id)
                    sequence.attention_mask = F.pad(sequence.attention_mask, (0, padding_length), value=0)
                input_ids_list.append(sequence.input_ids.unsqueeze(0))
                attention_mask_list.append(sequence.attention_mask.unsqueeze(0))
            else:
                input_ids_list.append(sequence.generated_tokens[-1:].unsqueeze(0))
                attention_mask_list.append(sequence.attention_mask[-1:].unsqueeze(0))
                past_key_values_list.append(sequence.kv_cache)
        
        self._model_inputs.update(input_ids_list, attention_mask_list, past_key_values_list)

    @property
    def model_inputs(self):
        # if self._model_inputs is None:
        self._preprocess_sequences()
        return self._model_inputs

    def update_sequences(self, logits, kv_caches):
        for i, sequence in enumerate(self.sequences):
            last_token_logits = logits[i, -1, :]
            next_token_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            
            # Extract and store the new KV cache only for the i-th sequence
            sequence_kv_cache = []
            for layer_idx, (key_layer, value_layer) in enumerate(kv_caches):
                # Extract the i-th sequence's cache for each layer
                key_i = key_layer[i].unsqueeze(0)  # Keep the batch dimension
                value_i = value_layer[i].unsqueeze(0)
                sequence_kv_cache.append((key_i, value_i))
            
            sequence.update(next_token_ids, sequence_kv_cache)
            if sequence.stage == "prefill":
                sequence.stage = "decode"