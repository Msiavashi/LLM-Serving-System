import queue
import torch
import time

from src.sequence import Sequence
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy
from src.performance_metrics import PerformanceMetrics

class Scheduler:
    def __init__(self, model, tokenizer, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_queue = SequenceQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size, self.sequence_queue)
        self.num_iterations = 100
        self.metrics = PerformanceMetrics()

    def add_sequence_to_queue(self, prompt, stage="prefill"):
        seq = Sequence(prompt, self.tokenizer, stage)
        self.sequence_queue.enqueue(seq)
         
    def run_scheduler(self):
        finished_sequences = []
        while not self.sequence_queue.is_empty():
            batch = self.batch_policy.get_next_batch()
             
            if batch.size() == 0:
                break
            
            with torch.no_grad():
                for i in range(self.num_iterations):
                    stage = "prefill" if i == 0 else "decode"
                    start_time = time.time()
                    outputs = self.model(batch=batch, use_cache=True)
                    end_time = time.time()
                    tokens_generated = sum(len(seq.input_ids) for seq in batch.sequences) + len(batch.sequences) if stage == "prefill" else len(outputs.sequences)
                    self.metrics.record_time(start_time, end_time, stage, tokens_generated=tokens_generated)
                    
            finished_sequences.extend(batch.sequences)
        
        self.metrics.report_metrics()
        return finished_sequences
