from .sequence_base import SequenceBase
import time

class Sequence(SequenceBase):
    def __init__(self, prompt, tokenizer, stage, device="cuda", priority: int=0, arrival_time=None):
        self.tokenizer = tokenizer
        self.cached_hidden_state = None
        self.routing_weights_cache = {}
        self.expert_outputs_cache = {}
        self.cached_residual = None
        self.stage = stage
        self.arrival_time = arrival_time or time.time()
        self.previous_token_time = self.arrival_time  # Initialize with arrival time
        self.finish_time = None
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        super().__init__(prompt, input_ids, attention_mask, kv_cache=None, device=device, priority=priority)