from src.batching.policies.base_policy import BaseBatchPolicy
from src.batching.batch import Batch

class SizeBasedBatchPolicy(BaseBatchPolicy):
    """
    A batch policy that creates batches based on a fixed size.

    Args:
        batch_size (int): The desired size of each batch.
        sequence_queue (Queue): A queue containing sequences to be batched.

    Attributes:
        batch_size (int): The desired size of each batch.
        sequence_queue (Queue): A queue containing sequences to be batched.
    """

    def __init__(self, batch_size, sequence_queue):
        self.batch_size = batch_size
        self.sequence_queue = sequence_queue
     
    def add_to_batch(self, sequence):
        # Not implemented
        pass
     
    def get_next_batch(self):
        """
        Retrieves the next batch of sequences.

        Returns:
            Batch: The next batch of sequences.
        """
        batch = Batch()
        while batch.size() < self.batch_size and not self.sequence_queue.is_empty():
            batch.add_sequence(self.sequence_queue.dequeue())
        return batch
