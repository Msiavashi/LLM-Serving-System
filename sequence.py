import torch
from typing import Literal

class SequenceBase:
    def __init__(self, prompt, input_ids, attention_mask, generated_tokens=None, kv_cache=None):
        self.prompt = prompt
        self.input_ids = input_ids.squeeze(0)  # Shape: [sequence_length]
        self.attention_mask = attention_mask.squeeze(0)  # Shape: [sequence_length]
        self.generated_tokens = generated_tokens if generated_tokens is not None else torch.empty(0, dtype=self.input_ids.dtype, device=self.input_ids.device)
        self.kv_cache = kv_cache
        self.stage: Literal["prefill", "decode"] = "prefill"

    def update(self, next_token_ids, new_kv_cache):
        next_token_ids = next_token_ids.to(self.generated_tokens.device)
        self.generated_tokens = torch.cat([self.generated_tokens, next_token_ids], dim=-1)
        self.attention_mask = torch.cat([self.attention_mask, torch.ones_like(next_token_ids, device=self.attention_mask.device)], dim=-1)
        self.kv_cache = new_kv_cache


    def get_generated_text(self, tokenizer):
        return tokenizer.decode(self.generated_tokens, skip_special_tokens=True)
    
    def __str__(self):
        return f"Sequence(prompt={self.prompt}, generated_text={self.generated_tokens})"
    
    def __repr__(self):
        return str(self)
       
class GenerationConfig:
    def __init__(self, max_tokens=10):
        self.max_tokens = max_tokens

class Sequence(SequenceBase):
    def __init__(self, prompt, tokenizer, generation_config: GenerationConfig = None, device="cuda"):
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        super().__init__(prompt, input_ids, attention_mask)
 