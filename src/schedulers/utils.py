import csv
import os

def write_csv_header(filename, header):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def write_queue_states(filename, iteration, queue_sizes):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [iteration] + [size for _, size in queue_sizes]
        writer.writerow(row)

def log_queue_sizes(engine, filename="./queue_states.csv"):
    header = []
    row = []
    
    for layer_idx, layer in enumerate(engine.model.model.layers):
        for queue_idx, queue in enumerate(layer.block_sparse_moe.queues):
            header.append(f"layer_{layer_idx}_expert_{queue_idx}")
            row.append(queue.size())
    
    if not os.path.exists(filename):
        write_csv_header(filename, header)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

class CleanOutputFormatter:
    """Formats output in a clean, vLLM-style format"""
    
    @staticmethod
    def print_request_summary(sequences, tokenizer):
        """Print a clean summary of completed requests"""
        if not sequences:
            return
            
        print("\n" + "="*80)
        print("REQUEST SUMMARY")
        print("="*80)
        
        total_requests = len(sequences)
        total_input_tokens = sum(seq.get_input_prompt_length() for seq in sequences)
        total_output_tokens = sum(seq.generated_tokens.numel() for seq in sequences)
        
        # Calculate timing metrics
        import time
        current_time = time.time()
        completion_times = []
        ttft_times = []
        
        for seq in sequences:
            if hasattr(seq, 'finish_time') and seq.finish_time:
                completion_times.append(seq.finish_time - seq.arrival_time)
            if hasattr(seq, 'first_token_time') and seq.first_token_time:
                ttft_times.append(seq.first_token_time - seq.arrival_time)
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        avg_ttft = sum(ttft_times) / len(ttft_times) if ttft_times else 0
        
        print(f"Total requests: {total_requests}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Average completion time: {avg_completion_time:.2f}s")
        print(f"Average time to first token: {avg_ttft:.2f}s")
        print("="*80)
    
    @staticmethod
    def print_request_result(seq, tokenizer, show_prompt=True):
        """Print a single request result in clean format"""
        generated_text = seq.get_generated_text(tokenizer)
        
        # Trim padding tokens
        if tokenizer.pad_token:
            generated_text = generated_text.replace(tokenizer.pad_token, "").strip()
        
        # Calculate metrics
        input_length = seq.get_input_prompt_length()
        output_length = seq.generated_tokens.numel()
        
        completion_time = 0
        ttft = 0
        if hasattr(seq, 'finish_time') and seq.finish_time and hasattr(seq, 'arrival_time'):
            completion_time = seq.finish_time - seq.arrival_time
        if hasattr(seq, 'first_token_time') and seq.first_token_time and hasattr(seq, 'arrival_time'):
            ttft = seq.first_token_time - seq.arrival_time
        
        print(f"\n{'─'*60}")
        print(f"Request ID: {seq.sequence_id}")
        if show_prompt:
            print(f"Prompt ({input_length} tokens): {seq.prompt}")
        print(f"Generated ({output_length} tokens): {generated_text}")
        if completion_time > 0:
            print(f"Completion time: {completion_time:.2f}s | TTFT: {ttft:.2f}s")
        print(f"{'─'*60}")
