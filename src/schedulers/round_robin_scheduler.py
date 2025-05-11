from typing import List
import torch
import time

from src.sequence import Sequence, Stage
from src.batching.policies import SizeBasedBatchPolicy
from .base_scheduler import BaseScheduler
from .model_instance import ModelInstance


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, models: List[ModelInstance], tokenizer, batch_size=32, rank=0):
        self.model_instances = models
        self.tokenizer = tokenizer
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        self.rank = rank

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
                
                # Update throughput stats
                elapsed = time.time() - start_time
                current_time = time.time()
                current_batch_latencies = []
                high_priority_latencies = []
                
                for seq in output_batch.sequences:
                    current_latency = current_time - seq.previous_token_time
                    current_batch_latencies.append(current_latency)
                    if seq.priority == 1:
                        high_priority_latencies.append(current_latency)
                    seq.previous_token_time = current_time
                    seq.sampling_metadata.current_token_count += 1
                    
                    if seq.sampling_metadata.current_token_count >= seq.sampling_metadata.max_sequence_length:
                        model_instance.finished_sequences.append(seq)
                        del seq.kv_cache
                    else:
                        model_instance.decode_queue.enqueue(seq)
                
                model_instance.monitor.record_batch(
                    is_decode=is_decode,
                    sequences=output_batch.sequences,
                    elapsed=elapsed,
                    sequence_latencies=current_batch_latencies,
                    high_priority_latencies=high_priority_latencies
                )
                
                # Print throughput for this iteration
                phase = "decode" if is_decode else "prefill"
                tokens_generated = sum(seq.get_total_sequence_length() if not is_decode else seq.sampling_metadata.current_token_count for seq in output_batch.sequences)
                print(f"Model {self.rank} - Iteration {iteration} ({phase}): "
                      f"Throughput = {tokens_generated/elapsed:.2f} tokens/sec "
                      f"Batch size = {tokens_generated} "
                      f"Elapsed time = {elapsed:.2f} sec")

    def run_scheduler(self):
        # Process single model instance
        self.process_model_instance(0, self.model_instances[0])
        
        finished_sequences = self.model_instances[0].finished_sequences

        # Print final statistics
        model_instance = self.model_instances[0]
        model_instance.monitor.print_final_stats()
        
        return finished_sequences
