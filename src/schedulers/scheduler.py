import torch
import time

from src.sequence import Sequence
from src.queues import FCFSQueue as SequenceQueue
from src.batching.policies import SizeBasedBatchPolicy

class Scheduler:
    def __init__(self, model, tokenizer, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.prefill_queue = SequenceQueue()
        self.decode_queue = SequenceQueue()
        self.batch_policy = SizeBasedBatchPolicy(batch_size)
        # Add throughput tracking
        self.prefill_stats = {"tokens": 0, "time": 0}
        self.decode_stats = {"tokens": 0, "time": 0}

    def add_sequence_to_queue(self, prompt, stage="prefill"):
        seq = Sequence(prompt, self.tokenizer, stage)
        if stage == "prefill":
            self.prefill_queue.enqueue(seq)
        elif stage == "decode":
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
            
            start_time = time.time()
            with torch.no_grad():
                output_batch = self.model(batch=batch, use_cache=True)
                tokens_generated = len(output_batch.sequences)
                
                # Update throughput stats
                elapsed = time.time() - start_time
                if is_decode:
                    self.decode_stats["tokens"] += tokens_generated
                    self.decode_stats["time"] += elapsed
                else:
                    self.prefill_stats["tokens"] += tokens_generated
                    self.prefill_stats["time"] += elapsed
                
                # Print throughput for this iteration
                phase = "decode" if is_decode else "prefill"
                print(f"Iteration {iteration} ({phase}): "
                      f"Throughput = {tokens_generated/elapsed:.2f} tokens/sec "
                      f"Batch size = {tokens_generated} "
                      f"Elapsed time = {elapsed:.2f} sec")
                
                for seq in output_batch.sequences:
                    seq.sampling_metadata.current_token_count += 1
                    if seq.sampling_metadata.current_token_count >= seq.sampling_metadata.max_sequence_length:
                        finished_sequences.append(seq)
                        del seq.kv_cache
                    else:
                        self.decode_queue.enqueue(seq)
        
        # Print final statistics
        if self.prefill_stats["time"] > 0:
            print(f"\nPrefill phase average throughput: "
                  f"{self.prefill_stats['tokens']/self.prefill_stats['time']:.2f} tokens/sec")
        if self.decode_stats["time"] > 0:
            print(f"Decode phase average throughput: "
                  f"{self.decode_stats['tokens']/self.decode_stats['time']:.2f} tokens/sec")
            
        return finished_sequences
