import time
import numpy as np
from statistics import median
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class PhaseStats:
    tokens: int = 0
    time: float = 0
    latencies: List[float] = field(default_factory=list)
    ttft_values: List[float] = field(default_factory=list)  # Track Time To First Token

class PerformanceMonitor:
    def __init__(self, measure_after_first_decode=False):
        self.prefill_stats = PhaseStats()
        self.decode_stats = PhaseStats()
        self.iteration = 0
        self.measure_after_first_decode = measure_after_first_decode
        self.first_decode_seen = False
        self.first_decode_time = None
        
    def record_batch(self, is_decode: bool, tokens_generated: int, elapsed: float, 
                    sequence_latencies: List[float]):
        # Track first successful decode
        if is_decode and tokens_generated > 0 and not self.first_decode_seen:
            self.first_decode_seen = True
            self.first_decode_time = time.time()
            if self.measure_after_first_decode:
                # Reset stats since we're starting measurements now
                self.prefill_stats = PhaseStats()
                self.decode_stats = PhaseStats()
        
        # Skip recording if waiting for first decode
        if self.measure_after_first_decode and not self.first_decode_seen:
            return
            
        stats = self.decode_stats if is_decode else self.prefill_stats
        stats.tokens += tokens_generated
        stats.time += elapsed # elapsed time of the iteration
        stats.latencies.extend(sequence_latencies) # elapsed time for each sequence compared to the previous token generation
        
        if not is_decode:  # First token (prefill phase)
            stats.ttft_values.extend(sequence_latencies)
        
        self.iteration += 1
        phase = "decode" if is_decode else "prefill"
        
        # Update print statements to distinguish TTFT and TPOT
        avg_tpot = np.mean(sequence_latencies) if sequence_latencies else 0
        med_tpot = median(sequence_latencies) if sequence_latencies else 0
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_time = self.prefill_stats.time + self.decode_stats.time
        
        if not is_decode:
            print(f"Iteration {self.iteration} ({phase}): "
                  f"Current Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Average Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Batch size = {tokens_generated}, "
                  f"TTFT = {avg_tpot:.3f} sec, "
                  f"Median TTFT = {med_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")
        else:
            print(f"Iteration {self.iteration} ({phase}): "
                  f"Current Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Average Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Batch size = {tokens_generated}, "
                  f"TPOT = {avg_tpot:.3f} sec, "
                  f"Median TPOT = {med_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")

    def print_final_stats(self):
        if self.measure_after_first_decode:
            print("\nNote: Measurements started after first successful decode")
            
        if self.prefill_stats.time > 0:
            avg_ttft = np.mean(self.prefill_stats.ttft_values) if self.prefill_stats.ttft_values else 0
            print(f"\nPrefill phase stats:"
                  f"\n  Average throughput: {self.prefill_stats.tokens/self.prefill_stats.time:.2f} tokens/sec"
                  f"\n  Average TTFT: {avg_ttft:.3f} sec")
        
        if self.decode_stats.time > 0:
            avg_tpot = np.mean(self.decode_stats.latencies) if self.decode_stats.latencies else 0
            print(f"\nDecode phase stats:"
                  f"\n  Average throughput: {self.decode_stats.tokens/self.decode_stats.time:.2f} tokens/sec"
                  f"\n  Average TPOT: {avg_tpot:.3f} sec")

        print(f"\nOverall TTFT: {avg_ttft:.3f} sec")
        print(f"Overall TPOT: {avg_tpot:.3f} sec")
        
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_duration = self.prefill_stats.time + self.decode_stats.time
        print(f"\nTotal throughput (all tokens/total duration): {total_tokens/total_duration:.2f} tokens/sec")
