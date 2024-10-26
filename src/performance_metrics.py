class PerformanceMetrics:
    def __init__(self):
        self.prefill_times = []
        self.decode_times = []
        self.prefill_token_count = 0
        self.decode_token_count = 0
    
    def record_time(self, start_time, end_time, stage, tokens_generated):
        elapsed_time = end_time - start_time
        if stage == "prefill":
            self.prefill_times.append(elapsed_time)
            self.prefill_token_count += tokens_generated
        elif stage == "decode":
            self.decode_times.append(elapsed_time)
            self.decode_token_count += tokens_generated

    def calculate_throughput(self, stage):
        total_time = sum(self.prefill_times if stage == "prefill" else self.decode_times)
        total_tokens = self.prefill_token_count if stage == "prefill" else self.decode_token_count
        return total_tokens / total_time if total_time > 0 else 0

    def calculate_latency(self, stage):
        times = self.prefill_times if stage == "prefill" else self.decode_times
        return sum(times) / len(times) if times else 0

    def report_metrics(self):
        print("Prefill Throughput (tokens/s):", self.calculate_throughput("prefill"))
        print("Prefill Latency (s):", self.calculate_latency("prefill"))
        print("Decode Throughput (tokens/s):", self.calculate_throughput("decode"))
        print("Decode Latency (s):", self.calculate_latency("decode"))

