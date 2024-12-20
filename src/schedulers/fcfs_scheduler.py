import torch
import time
from typing import List

from src.sequence import Sequence, Stage
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy
from .base_scheduler import BaseScheduler

class FCFSScheduler(BaseScheduler):
    def __init__(self, engine, tokenizer, batch_size=32):
        self.engine = engine
        self.tokenizer = tokenizer
        self.prefill_queue = SequenceQueue()
        self.decode_queue = SequenceQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.prefill_stats = {"tokens": 0, "time": 0}
        self.decode_stats = {"tokens": 0, "time": 0}

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL):
        seq = Sequence(prompt, self.tokenizer, stage)
        if stage == Stage.PREFILL:
            self.prefill_queue.enqueue(seq)
        elif stage == Stage.DECODE:
            self.decode_queue.enqueue(seq)

    def run_scheduler(self):
        finished_sequences = []
        iteration = 0
        
        while not (self.decode_queue.is_empty() and self.prefill_queue.is_empty()):
            iteration += 1
            is_decode = not self.decode_queue.is_empty()
            
            if is_decode:
                batch = self.batch_policy.get_next_batch(self.decode_queue)
            else:
                batch = self.batch_policy.get_next_batch(self.prefill_queue)
             
            if batch.size() == 0:
                break
            
            sequences = self.engine.run_batch(batch)
            
            for seq in sequences:
                seq.sampling_metadata.current_token_count += 1
                if seq.sampling_metadata.current_token_count >= seq.sampling_metadata.max_sequence_length:
                    finished_sequences.append(seq)
                    del seq.kv_cache
                else:
                    self.decode_queue.enqueue(seq)
            
        return finished_sequences