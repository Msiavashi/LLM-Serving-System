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

class PerformanceMonitor:
    def __init__(self):
        self.prefill_stats = PhaseStats()
        self.decode_stats = PhaseStats()
        self.iteration = 0
        
    def record_batch(self, is_decode: bool, tokens_generated: int, elapsed: float, 
                    sequence_latencies: List[float]):
        stats = self.decode_stats if is_decode else self.prefill_stats
        stats.tokens += tokens_generated
        stats.time += elapsed
        stats.latencies.extend(sequence_latencies)
        
        self.iteration += 1
        phase = "decode" if is_decode else "prefill"
        
        # Print iteration stats
        avg_latency = np.mean(sequence_latencies) if sequence_latencies else 0
        med_latency = median(sequence_latencies) if sequence_latencies else 0
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_time = self.prefill_stats.time + self.decode_stats.time
        
        print(f"Iteration {self.iteration} ({phase}): "
              f"Current Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
              f"Average Throughput = {total_tokens/total_time:.2f} tokens/sec, "
              f"Batch size = {tokens_generated}, "
              f"Avg Latency = {avg_latency:.3f} sec, "
              f"Median Latency = {med_latency:.3f} sec, "
              f"Elapsed time = {elapsed:.2f} sec")

    def print_final_stats(self):
        if self.prefill_stats.time > 0:
            avg_prefill_latency = np.mean(self.prefill_stats.latencies) if self.prefill_stats.latencies else 0
            print(f"\nPrefill phase stats:"
                  f"\n  Average throughput: {self.prefill_stats.tokens/self.prefill_stats.time:.2f} tokens/sec"
                  f"\n  Average latency: {avg_prefill_latency:.3f} sec")
        
        if self.decode_stats.time > 0:
            avg_decode_latency = np.mean(self.decode_stats.latencies) if self.decode_stats.latencies else 0
            print(f"\nDecode phase stats:"
                  f"\n  Average throughput: {self.decode_stats.tokens/self.decode_stats.time:.2f} tokens/sec"
                  f"\n  Average latency: {avg_decode_latency:.3f} sec")

        combined_latencies = self.prefill_stats.latencies + self.decode_stats.latencies
        overall_avg_latency = np.mean(combined_latencies) if combined_latencies else 0
        print(f"\nOverall average latency: {overall_avg_latency:.3f} sec")
        
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_duration = self.prefill_stats.time + self.decode_stats.time
        print(f"\nTotal throughput (all tokens/total duration): {total_tokens/total_duration:.2f} tokens/sec")
