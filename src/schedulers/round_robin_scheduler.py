from typing import List
import torch
import time
import threading
from mpi4py import MPI

from src.sequence import Sequence, Stage
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy
from .base_scheduler import BaseScheduler

class ModelInstance:
    def __init__(self, model, device):
        self.model = model  # Don't move the model
        self.device = device  # Just store target device for sequences
        self.prefill_queue = SequenceQueue()
        self.decode_queue = SequenceQueue()
        self.prefill_stats = {"tokens": 0, "time": 0}
        self.decode_stats = {"tokens": 0, "time": 0}
        self.finished_sequences = []

class RoundRobinScheduler(BaseScheduler):
    def __init__(self, models: List[ModelInstance], tokenizer, batch_size=32):
        self.model_instances = models
        self.tokenizer = tokenizer
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.rank = MPI.COMM_WORLD.Get_rank()

    def add_sequence_to_queue(self, prompt, stage=Stage.PREFILL):
        # Single model per rank, so always use first model instance
        model_instance = self.model_instances[0]
        seq = Sequence(prompt, self.tokenizer, stage, device=model_instance.device)
        
        if stage == Stage.PREFILL:
            model_instance.prefill_queue.enqueue(seq)
        else:
            model_instance.decode_queue.enqueue(seq)

    def process_model_instance(self, model_idx, model_instance):
        iteration = 0
        while not (model_instance.decode_queue.is_empty() and model_instance.prefill_queue.is_empty()):
            iteration += 1
            
            is_decode = not model_instance.decode_queue.is_empty()
            queue = model_instance.decode_queue if is_decode else model_instance.prefill_queue
            batch = self.batch_policy.get_next_batch(queue)
            
            if batch.size() == 0:
                continue
                
            start_time = time.time()
            with torch.no_grad():
                output_batch = model_instance.model(batch=batch, use_cache=True)
                tokens_generated = len(output_batch.sequences)
                
                elapsed = time.time() - start_time
                stats = model_instance.decode_stats if is_decode else model_instance.prefill_stats
                stats["tokens"] += tokens_generated
                stats["time"] += elapsed
                
                phase = "decode" if is_decode else "prefill"
                print(f"Rank/Model {self.rank} - Iteration {iteration} ({phase}): "
                        f"Throughput = {tokens_generated/elapsed:.2f} tokens/sec "
                        f"Batch size = {tokens_generated} "
                        f"Elapsed time = {elapsed:.2f} sec")
                
                for seq in output_batch.sequences:
                    seq.sampling_metadata.current_token_count += 1
                    if seq.sampling_metadata.current_token_count >= seq.sampling_metadata.max_sequence_length:
                        model_instance.finished_sequences.append(seq)
                        del seq.kv_cache
                    else:
                        model_instance.decode_queue.enqueue(seq)

    def run_scheduler(self):
        # Process single model instance
        self.process_model_instance(0, self.model_instances[0])
        
        finished_sequences = self.model_instances[0].finished_sequences

        # Print statistics with rank
        model_instance = self.model_instances[0]
        print(f"\nRank/Model {self.rank} Statistics:")
        if model_instance.prefill_stats["time"] > 0:
            print(f"Prefill phase average throughput: "
                  f"{model_instance.prefill_stats['tokens']/model_instance.prefill_stats['time']:.2f} tokens/sec")
        if model_instance.decode_stats["time"] > 0:
            print(f"Decode phase average throughput: "
                  f"{model_instance.decode_stats['tokens']/model_instance.decode_stats['time']:.2f} tokens/sec")
        
        return finished_sequences
