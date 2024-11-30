from src.batching.batch import Batch


class ModelOutputMixin:
    """
    Mixin to handle model outputs and update batch accordingly.
    """
    def _update_batch(self, outputs, running_batch: Batch):
        logits = outputs.logits
        kv_cache = outputs.past_key_values

        # Splitting key-value cache for update
        split_kv_cache = kv_cache.split_kv_cache()

        # Updating the running batch with new sequences and split key-value cache
        running_batch.update_sequences(logits, split_kv_cache)