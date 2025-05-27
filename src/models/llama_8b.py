from transformers.models.llama.modeling_llama import LlamaForCausalLM
from src.batching.batch import Batch


class Llama8B(LlamaForCausalLM):
    def forward(self, input_ids, attention_mask, past_key_values, running_batch: Batch = None, **kwargs) -> Batch:
        # input_ids, attention_mask, past_key_values, running_batch are prepared by the strategy
        outputs = super().forward(input_ids, attention_mask=None, past_key_values=past_key_values, **kwargs)
        # The strategy will handle updating the batch with outputs
        return outputs
