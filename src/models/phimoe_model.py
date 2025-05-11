from transformers import PhimoeForCausalLM
from src.mixins.model_input_mixin import ModelInputMixin
from src.mixins.model_output_mixin import ModelOutputMixin
from src.batching.batch import Batch

class PhiMoe(PhimoeForCausalLM, ModelInputMixin, ModelOutputMixin):
    def forward(self, batch: Batch, **kwargs) -> Batch:
        # Prepare inputs
        input_ids, attention_mask, past_key_values, running_batch = self._prepare_inputs(batch)
 
        outputs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        
        self._update_batch(outputs, running_batch)

        return self.running_batch
    