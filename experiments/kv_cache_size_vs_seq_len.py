import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
# from src.models.mixtral_model import MyCustomMixtral
from src.models.mixtral_queue_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler
import random


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
        attn_implementation="flash_attention_2",
    )
    
    return model, tokenizer


def create_large_prompts(length=128*1024, num_prompts=32):
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
    
    return [current_text] * num_prompts

def usage_example():
    model, tokenizer = initialize_model_and_tokenizer()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6} MB")
    scheduler = Scheduler(model, tokenizer)

    # Test specific lengths
    # test_lengths = [32*1024, 64*1024, 96*1024, 128*1024, 160*1024, 192*1024]
    test_lengths = [int(math.fabs(5.9*1024))]
    num_prompts = 32  # Number of prompts to generate
    max_successful_length = 0

    for length in test_lengths:
        print(f"\nTesting prompt length: {length} tokens")
        torch.cuda.empty_cache()  # Clear GPU memory before each test
        
        try:
            prompts = create_large_prompts(length, num_prompts)
            for prompt in prompts:
                scheduler.add_sequence_to_queue(prompt)
            results = scheduler.run_scheduler()
            max_successful_length = length
            print(f"✓ Successfully processed {num_prompts} prompts of length {length}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ Memory error at length {length}")
                break
            else:
                print(f"✗ Unexpected error at length {length}: {str(e)}")
                break
    
    print(f"\nMaximum successful length: {max_successful_length}")

if __name__ == "__main__":
    usage_example()