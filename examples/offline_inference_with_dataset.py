import sys
import os
import argparse
from tqdm import tqdm
import concurrent.futures
import torch

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
# from src.models.mixtral_queue_model import MyCustomMixtral
from src.models.mixtral_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler
from utils import read_shared_gpt_dataset

def create_large_prompt(length=192*1024):
    base_text = "Generate a detailed response. "
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    def get_token_length(text):
        return len(tokenizer.encode(text))
    
    # Binary search to find the correct text length
    target_length = length
    current_text = base_text
    left = 1
    right = length * 2
    
    while get_token_length(current_text) != target_length:
        multiplier = (left + right) // 2
        current_text = base_text * multiplier
        current_length = get_token_length(current_text)
        
        if current_length < target_length:
            left = multiplier + 1
        elif current_length > target_length:
            right = multiplier - 1
        
        if left > right:
            while get_token_length(current_text) > target_length:
                current_text = current_text[:-1]
            while get_token_length(current_text) < target_length:
                current_text += base_text[0]
            break
    
    return current_text

def generate_prompts(seq_len, num_prompts):
    """Generate multiple prompts with specified length using multi-threading."""
    def generate_single_prompt(_):
        return create_large_prompt(seq_len)
    
    prompts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_single_prompt, i) for i in range(num_prompts)]
        
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=num_prompts,
            desc="Generating prompts"
        ):
            prompts.append(future.result())
    
    return prompts

def prepare_prompts(args, dataset_path):
    """Prepare prompts either from dataset or generate them."""
    if args.use_dataset and os.path.exists(dataset_path):
        return read_shared_gpt_dataset(dataset_path, args.num_prompts)
    elif args.generate:
        return generate_prompts(args.seq_len, args.num_prompts)
    else:
        return [create_large_prompt()]  # default case

def initialize_model_and_tokenizer():
    config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = MyCustomMixtral.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        config=config,
        device_map='auto',
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",  # Enable flash attention
    )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run inference with Mixtral model using queue-based scheduling")
    parser.add_argument('--use-dataset', action='store_true', help='Use dataset for prompts')
    parser.add_argument('--generate', action='store_true', help='Generate prompts with specified length')
    parser.add_argument('--seq-len', type=int, default=192*1024, help='Sequence length for generated prompts')
    parser.add_argument('--num-prompts', type=int, default=1, help='Number of prompts to process')
    args = parser.parse_args()

    dataset_path = "./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    model, tokenizer = initialize_model_and_tokenizer()
    scheduler = Scheduler(model, tokenizer)
    
    prompts = prepare_prompts(args, dataset_path)
    
    # Add sequences to scheduler queue
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    # Run scheduler and get results
    results = scheduler.run_scheduler()

    # Process results
    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Input length: {len(tokenizer.encode(seq.prompt))}")
        print(f"Generated: {generated_text}\n")

if __name__ == "__main__":
    main()