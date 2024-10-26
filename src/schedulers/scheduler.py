import queue
import torch

from src.sequence import Sequence
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy

class Scheduler:
    def __init__(self, model, tokenizer, batch_size=5):
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_queue = SequenceQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size, self.sequence_queue)
        self.num_iterations = 10

    def add_sequence_to_queue(self, prompt, stage="prefill"):
        seq = Sequence(prompt, self.tokenizer, stage)
        self.sequence_queue.enqueue(seq)
         
    def run_scheduler(self):
        finished_sequences = []
        while not self.sequence_queue.is_empty():
            batch: Batch = self.batch_policy.get_next_batch()
             
            if batch.size() == 0:
                break
            
            with torch.no_grad():
                for i in range(self.num_iterations):
                    outputs = self.model(batch=batch, use_cache=True)
            finished_sequences.extend(batch.sequences)
        return finished_sequences