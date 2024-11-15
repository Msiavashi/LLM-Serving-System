import torch
import time

from src.sequence import Sequence
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy

class Scheduler:
    def __init__(self, model, tokenizer, batch_size=32):
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
        seen_sequences = set()
        
        while not self.sequence_queue.is_empty():
            batch = self.batch_policy.get_next_batch()
             
            if batch.size() == 0:
                break
            
            with torch.no_grad():
                for i in range(self.num_iterations):
                    start_time = time.time()
                    output_batch = self.model(batch=batch, use_cache=True)
                    print(f"Output tokens: {output_batch.size()}")
                    end_time = time.time()
                    
                    tokens_generated = len(output_batch.sequences)
                    elapsed_time = end_time - start_time
                    throughput = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                    print(f"Throughput (tokens/s): {throughput}")
                    
                    for seq in output_batch.sequences:
                        seq_id = id(seq)
                        if seq_id not in seen_sequences:
                            seen_sequences.add(seq_id)
                            finished_sequences.append(seq)
                    
                    if batch.size() == 0:
                        break
        
        return finished_sequences
