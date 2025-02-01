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
    high_priority_latencies: List[float] = field(default_factory=list)
    ttft_values: List[float] = field(default_factory=list)  # Track Time To First Token

class PerformanceMonitor:
    def __init__(self):
        self.prefill_stats = PhaseStats()
        self.decode_stats = PhaseStats()
        self.iteration = 0
        
    def record_batch(self, is_decode: bool, tokens_generated: int, elapsed: float, 
                    sequence_latencies: List[float], high_priority_latencies: List[float]):
        stats = self.decode_stats if is_decode else self.prefill_stats
        stats.tokens += tokens_generated
        stats.time += elapsed
        stats.latencies.extend(sequence_latencies)
        stats.high_priority_latencies.extend(high_priority_latencies)
        
        if not is_decode:  # First token (prefill phase)
            stats.ttft_values.extend(sequence_latencies)
        
        self.iteration += 1
        phase = "decode" if is_decode else "prefill"
        
        # Update print statements to distinguish TTFT and TPOT
        avg_tpot = np.mean(sequence_latencies) if sequence_latencies else 0
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_time = self.prefill_stats.time + self.decode_stats.time
        
        print("--" * 40)
        if not is_decode:
            print(f"Iteration {self.iteration} ({phase}): "
                  f"Iteration Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Cumulative Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Output size = {tokens_generated}, "
                  f"Iteration TTFT = {avg_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")
        else:
            print(f"Iteration {self.iteration} ({phase}): "
                  f"Iteration Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Cumulative Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Output size = {tokens_generated}, "
                  f"Iteration TPOT = {avg_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")

        if high_priority_latencies:
            avg_high_priority_latency = np.mean(high_priority_latencies) if high_priority_latencies else 0
            p90_high_priority_latency = np.percentile(high_priority_latencies, 90) if high_priority_latencies else 0
            print(f"LS Avg Latency = {avg_high_priority_latency:.3f} sec, "
                  f"LS P90 Latency = {p90_high_priority_latency:.3f} sec")
            # Also print overall average latency
            overal_avg_latency = np.mean(stats.high_priority_latencies) if stats.high_priority_latencies else 0
            overal_p90_high_priority_latency = np.percentile(stats.high_priority_latencies, 90) if stats.high_priority_latencies else 0 
            print(f"LS Overall Avg Latency = {overal_avg_latency:.3f} sec, "
                    f"LS Overall P90 Latency = {overal_p90_high_priority_latency:.3f} sec")
        # print total time elapsed and total tokens generated
        print(f"Total time elapsed = {total_time:.2f} sec, "
                f"Total tokens generated = {stats.tokens}")
            

    def print_final_stats(self):
        avg_ttft = 0
        avg_tpot = 0
        print("--" * 40)
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

        combined_latencies = self.prefill_stats.latencies + self.decode_stats.latencies
        overall_avg_latency = np.mean(combined_latencies) if combined_latencies else 0
        print(f"\nOverall average latency: {overall_avg_latency:.3f} sec")
        
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_duration = self.prefill_stats.time + self.decode_stats.time
        print(f"\nTotal throughput (all tokens/total duration): {total_tokens/total_duration:.2f} tokens/sec")
        
        if self.prefill_stats.high_priority_latencies:
            avg_prefill_high_priority_latency = np.mean(self.prefill_stats.high_priority_latencies)
            p90_prefill_high_priority_latency = np.percentile(self.prefill_stats.high_priority_latencies, 90)
            p99_prefill_high_priority_latency = np.percentile(self.prefill_stats.high_priority_latencies, 99)
            print(f"\nPrefill High Priority Latency stats:"
              f"\n  Average Latency: {avg_prefill_high_priority_latency:.3f} sec"
              f"\n  P90 Latency: {p90_prefill_high_priority_latency:.3f} sec"
              f"\n  P99 Latency: {p99_prefill_high_priority_latency:.3f} sec")

        if self.decode_stats.high_priority_latencies:
            avg_decode_high_priority_latency = np.mean(self.decode_stats.high_priority_latencies)
            p90_decode_high_priority_latency = np.percentile(self.decode_stats.high_priority_latencies, 90)
            p99_decode_high_priority_latency = np.percentile(self.decode_stats.high_priority_latencies, 99)
            print(f"\nDecode High Priority Latency stats:"
              f"\n  Average Latency: {avg_decode_high_priority_latency:.3f} sec"
              f"\n  P90 Latency: {p90_decode_high_priority_latency:.3f} sec"
              f"\n  P99 Latency: {p99_decode_high_priority_latency:.3f} sec")

        print(f"\nOverall TTFT: {avg_ttft:.3f} sec")
        print(f"Overall TPOT: {avg_tpot:.3f} sec")