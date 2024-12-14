import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import argparse
from utils import read_shared_gpt_dataset
from transformers.cache_utils import DynamicCache
import concurrent.futures
import time  # Add at the top with other imports


def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    if (tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device_map='auto',
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        offload_folder=None,  # Prevent disk offloading
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2",  # Enable flash attention
    )
    
    return model, tokenizer

def create_large_prompt(length=192*1024):
    base_text = "Generate a detailed response. "
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    def get_token_length(text):
        return len(tokenizer.encode(text))
    
    # Binary search to find the correct text length
    target_length = length
    current_text = base_text
    left = 1
    right = length * 2  # Overestimate to ensure we can reach target length
    
    while get_token_length(current_text) != target_length:
        multiplier = (left + right) // 2
        current_text = base_text * multiplier
        current_length = get_token_length(current_text)
        
        if current_length < target_length:
            left = multiplier + 1
        elif current_length > target_length:
            right = multiplier - 1
        
        # Prevent infinite loop if exact length cannot be achieved
        if left > right:
            # Fine-tune by adding or removing individual characters
            while get_token_length(current_text) > target_length:
                current_text = current_text[:-1]
            while get_token_length(current_text) < target_length:
                current_text += base_text[0]
            break
    
    return current_text


def calculate_dynamic_cache_size_in_mb(dynamic_cache):
    total_size_bytes = 0

    # Iterate over each layer's key and value caches
    for layer_idx in range(len(dynamic_cache.key_cache)):
        key_tensor = dynamic_cache.key_cache[layer_idx]
        value_tensor = dynamic_cache.value_cache[layer_idx]

        if key_tensor is not None:
            total_size_bytes += key_tensor.element_size() * key_tensor.nelement()
        if value_tensor is not None:
            total_size_bytes += value_tensor.element_size() * value_tensor.nelement()

    # Convert bytes to megabytes
    total_size_mb = total_size_bytes / (1024 ** 2)
    return total_size_mb


def single_inference(prompts, model, tokenizer, output_length=1):
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = inputs["input_ids"].shape[1]
    total_tokens = input_length * inputs["input_ids"].shape[0]
    print(f"Batch size: {inputs['input_ids'].shape[0]}, Total tokens: {total_tokens}")
    
    # Initialize DynamicCache
    dynamic_cache = DynamicCache()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=output_length,
            min_new_tokens=output_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            past_key_values=dynamic_cache,
            num_return_sequences=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        kv_cache_size_mb = calculate_dynamic_cache_size_in_mb(outputs.past_key_values)
        
        print(f"KV cache size: {kv_cache_size_mb:.2f} MB")
    
    new_tokens = outputs.sequences[:, input_length:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

def generate_prompts(seq_len, num_prompts):
    """Generate one prompt and duplicate it."""
    single_prompt = create_large_prompt(seq_len)
    return [single_prompt] * num_prompts


def prepare_prompts(args, dataset_path):
    """Prepare prompts either from dataset or generate them."""
    if args.use_dataset and os.path.exists(dataset_path):
        return read_shared_gpt_dataset(dataset_path, args.num_prompts)
    elif args.generate:
        return generate_prompts(args.seq_len, args.num_prompts)
    else:
        return [create_large_prompt()]  # default case

def batch_inference(prompts, model, tokenizer, batch_size, output_length):
    """Process prompts in batches."""
    results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size  # ceiling division
    progress_bar = tqdm(total=num_batches, desc="Processing batches")
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Calculate total input tokens for this batch
        total_input_tokens = sum(len(tokenizer.encode(p)) for p in batch)
        total_tokens = total_input_tokens + (len(batch) * output_length)
        
        # Measure time
        start_time = time.time()
        outputs = single_inference(batch, model, tokenizer, output_length)
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = batch_size / elapsed_time
        
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tokens/second")
        
        for j, output in enumerate(outputs):
            del output
        
        progress_bar.update(1)
    
    progress_bar.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with Mixtral model")
    parser.add_argument('--use-dataset', action='store_true', help='Use dataset for prompts')
    parser.add_argument('--generate', action='store_true', help='Generate prompts with specified length')
    parser.add_argument('--seq-len', type=int, default=192*1024, help='Sequence length for generated prompts')
    parser.add_argument('--num-prompts', type=int, default=1, help='Number of prompts to process')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output-length', type=int, default=1, help='Number of tokens to generate')
    args = parser.parse_args()

    dataset_path = "./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    model, tokenizer = initialize_model_and_tokenizer()
    
    prompts = prepare_prompts(args, dataset_path)
    results = batch_inference(prompts, model, tokenizer, args.batch_size, args.output_length)

    # Print results
    # for result in results:
    #     print(f"Prompt length: {result['prompt_length']}")
        # print(f"Generated: {result['output']}\n")

if __name__ == "__main__":
    main()