import logging
import time
import numpy as np
from statistics import median
from typing import List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PhaseStats:
    tokens: int = 0
    time: float = 0
    latencies: List[float] = field(default_factory=list)
    high_priority_latencies: List[float] = field(default_factory=list)
    ttft_values: List[float] = field(default_factory=list)  # Track Time To First Token
    # Removed finished_jobs and finished_high_priority_jobs from PhaseStats

class PerformanceMonitor:
    def __init__(self):
        self.prefill_stats = PhaseStats()
        self.decode_stats = PhaseStats()
        self.iteration = 0
        # Global job metrics (not assigned to a phase)
        self.finished_jobs = 0
        self.finished_hp_jobs = 0
        self.turnaround_times = []     # List[float] for all sequences
        self.hp_turnaround_times = []  # List[float] for high priority sequences

    def record_batch(self, is_decode: bool, sequences: List, elapsed: float, 
                    sequence_latencies: List[float], high_priority_latencies: List[float]):
        stats = self.decode_stats if is_decode else self.prefill_stats
        tokens_generated = len(sequences) if is_decode else sum(seq.get_total_sequence_length() for seq in sequences)
        stats.tokens += tokens_generated
        stats.time += elapsed
        stats.latencies.extend(sequence_latencies)
        stats.high_priority_latencies.extend(high_priority_latencies)
        
        if not is_decode:  # First token (prefill phase)
            stats.ttft_values.extend(sequence_latencies)
        
        # Count finished jobs using finish_time field and update global counters
        finished = sum(1 for seq in sequences if hasattr(seq, 'finish_time') and seq.finish_time is not None)
        hp_finished = sum(1 for seq in sequences if hasattr(seq, 'finish_time') and seq.finish_time is not None and seq.priority == 1)
        self.finished_jobs += finished
        self.finished_hp_jobs += hp_finished
        
        # Compute turnaround times from arrival to finish for finished sequences
        turnaround_times = [seq.finish_time - seq.arrival_time for seq in sequences if hasattr(seq, 'finish_time') and seq.finish_time is not None and hasattr(seq, 'arrival_time')]
        if turnaround_times:
            self.turnaround_times.extend(turnaround_times)
            avg_turnaround = np.mean(turnaround_times)
        else:
            avg_turnaround = 0
        
        # Compute high priority turnaround times
        hp_turnaround_times = [seq.finish_time - seq.arrival_time for seq in sequences if hasattr(seq, 'finish_time') and seq.finish_time is not None and hasattr(seq, 'arrival_time') and seq.priority == 1]
        if hp_turnaround_times:
            self.hp_turnaround_times.extend(hp_turnaround_times)
            avg_hp_turnaround = np.mean(hp_turnaround_times)
        else:
            avg_hp_turnaround = 0

        self.iteration += 1
        phase = "decode" if is_decode else "prefill"
        
        avg_tpot = np.mean(sequence_latencies) if sequence_latencies else 0
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_time = self.prefill_stats.time + self.decode_stats.time
        
        logger.info("--" * 40)
        if not is_decode:
            logger.info(f"Iteration {self.iteration} ({phase}): "
                  f"Iteration Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Cumulative Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Output size = {tokens_generated}, "
                  f"Iteration TTFT = {avg_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")
        else:
            logger.info(f"Iteration {self.iteration} ({phase}): "
                  f"Iteration Throughput = {tokens_generated/elapsed:.2f} tokens/sec, "
                  f"Cumulative Throughput = {total_tokens/total_time:.2f} tokens/sec, "
                  f"Output size = {tokens_generated}, "
                  f"Iteration TPOT = {avg_tpot:.3f} sec, "
                  f"Elapsed time = {elapsed:.2f} sec")

        if high_priority_latencies:
            avg_high_priority_latency = np.mean(high_priority_latencies)
            p90_high_priority_latency = np.percentile(high_priority_latencies, 90)
            logger.info(f"Iteration LS Avg Latency = {avg_high_priority_latency:.3f} sec, "
                  f"Iteration LS P90 Latency = {p90_high_priority_latency:.3f} sec")
            overal_avg_latency = np.mean(stats.high_priority_latencies) if stats.high_priority_latencies else 0
            overal_p90_high_priority_latency = np.percentile(stats.high_priority_latencies, 90) if stats.high_priority_latencies else 0 
            logger.info(f"{phase} LS Overall Avg Latency = {overal_avg_latency:.3f} sec, "
                  f"{phase} LS Overall P90 Latency = {overal_p90_high_priority_latency:.3f} sec")
            
        # Compute global job completion per second
        global_job_comp = self.finished_jobs / total_time if total_time > 0 else 0
        global_hp_job_comp = self.finished_hp_jobs / total_time if total_time > 0 else 0
        logger.info(f"Job completion: {global_job_comp:.2f} jobs/sec")
        logger.info(f"LS Job completion: {global_hp_job_comp:.2f} jobs/sec")
        
        # Print average turnaround time computed from arrival and finish times
        if turnaround_times:
            logger.info(f"Avg Turnaround Time: {avg_turnaround:.3f} sec")
            if hp_turnaround_times:
                logger.info(f"LS Avg Turnaround Time: {avg_hp_turnaround:.3f} sec")
        
        logger.info(f"Total time elapsed = {total_time:.2f} sec, "
              f"Total tokens generated = {stats.tokens}")
        logger.info()

    def print_final_stats(self):
        if self.prefill_stats.time == 0 and self.decode_stats.time == 0:
            return
        
        avg_ttft = 0
        avg_tpot = 0
        logger.info("\n" + "="*80)
        if self.prefill_stats.time > 0:
            avg_ttft = np.mean(self.prefill_stats.ttft_values) if self.prefill_stats.ttft_values else 0
            logger.info(f"\nPrefill phase stats:"
                  f"\n  Average throughput: {self.prefill_stats.tokens/self.prefill_stats.time:.2f} tokens/sec"
                  f"\n  Average TTFT: {avg_ttft:.3f} sec")
        
        if self.decode_stats.time > 0:
            avg_tpot = np.mean(self.decode_stats.latencies) if self.decode_stats.latencies else 0
            logger.info(f"\nDecode phase stats:"
                  f"\n  Average throughput: {self.decode_stats.tokens/self.decode_stats.time:.2f} tokens/sec"
                  f"\n  Average TPOT: {avg_tpot:.3f} sec")
        
        combined_latencies = self.prefill_stats.latencies + self.decode_stats.latencies
        overall_avg_latency = np.mean(combined_latencies) if combined_latencies else 0
        logger.info(f"\nOverall average latency: {overall_avg_latency:.3f} sec")
        
        total_tokens = self.prefill_stats.tokens + self.decode_stats.tokens
        total_duration = self.prefill_stats.time + self.decode_stats.time
        if total_duration > 0:
            logger.info(f"\nTotal throughput (all tokens/total duration): {total_tokens/total_duration:.2f} tokens/sec")
        
        # Use global job counters for overall job metrics
        overall_jobs_rate = self.finished_jobs / total_duration if total_duration > 0 else 0
        overall_hp_jobs_rate = self.finished_hp_jobs / total_duration if total_duration > 0 else 0
        
        logger.info(f"\nOverall job completion rate: {overall_jobs_rate:.2f} jobs/sec"
              f"\nOverall LS job completion rate: {overall_hp_jobs_rate:.2f} jobs/sec")
        
        if self.prefill_stats.high_priority_latencies:
            avg_prefill_high_priority_latency = np.mean(self.prefill_stats.high_priority_latencies)
            p90_prefill_high_priority_latency = np.percentile(self.prefill_stats.high_priority_latencies, 90)
            p99_prefill_high_priority_latency = np.percentile(self.prefill_stats.high_priority_latencies, 99)
            logger.info(f"\nPrefill High Priority Latency stats:"
                  f"\n  Average Latency: {avg_prefill_high_priority_latency:.3f} sec"
                  f"\n  P90 Latency: {p90_prefill_high_priority_latency:.3f} sec"
                  f"\n  P99 Latency: {p99_prefill_high_priority_latency:.3f} sec")

        if self.decode_stats.high_priority_latencies:
            avg_decode_high_priority_latency = np.mean(self.decode_stats.high_priority_latencies)
            p90_decode_high_priority_latency = np.percentile(self.decode_stats.high_priority_latencies, 90)
            p99_decode_high_priority_latency = np.percentile(self.decode_stats.high_priority_latencies, 99)
            logger.info(f"\nDecode High Priority Latency stats:"
                  f"\n  Average Latency: {avg_decode_high_priority_latency:.3f} sec"
                  f"\n  P90 Latency: {p90_decode_high_priority_latency:.3f} sec"
                  f"\n  P99 Latency: {p99_decode_high_priority_latency:.3f} sec")

        combined_high_priority_latencies = self.prefill_stats.high_priority_latencies + self.decode_stats.high_priority_latencies
        overall_avg_high_priority_latency = np.mean(combined_high_priority_latencies) if combined_high_priority_latencies else 0
        logger.info(f"\nOverall average high priority latency: {overall_avg_high_priority_latency:.3f} sec")
        logger.info(f"\nOverall TTFT: {avg_ttft:.3f} sec")
        logger.info(f"Overall TPOT: {avg_tpot:.3f} sec")
        # Print overall turnaround metrics computed from global arrays
        if self.turnaround_times:
            overall_avg_turnaround = np.mean(self.turnaround_times)
            logger.info(f"\nOverall Avg Turnaround Time: {overall_avg_turnaround:.3f} sec")
        if self.hp_turnaround_times:
            overall_hp_avg_turnaround = np.mean(self.hp_turnaround_times)
            logger.info(f"Overall LS Avg Turnaround Time: {overall_hp_avg_turnaround:.3f} sec")
        logger.info("="*80 + "\n")