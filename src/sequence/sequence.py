from .sequence_base import SequenceBase
import time
from src.samplers.sampling_metadata import SamplingMetadata

class Sequence(SequenceBase):
    def __init__(self, prompt, tokenizer, stage, cache_provider, device="cuda", sampling_metadata=None):
        self.tokenizer = tokenizer
        self.cached_hidden_state = None
        self.routing_weights_cache = {}
        self.expert_outputs_cache = {}
        self.cached_residual = None
        self.stage = stage
        self.arrival_time = time.time()
        self.previous_token_time = self.arrival_time  # Initialize with arrival time
        self.finish_time = None
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        kv_cache = cache_provider.create_sequence_cache(batch_size=1, max_sequence_length=100)
        sampling_metadata = sampling_metadata if sampling_metadata is not None else SamplingMetadata(num_tokens=5)
        super().__init__(prompt, input_ids, attention_mask, kv_cache=kv_cache, device=device, sampling_metadata=sampling_metadata)
        