import sys
import os

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
# from src.models.mixtral_model import MyCustomMixtral
from src.models.mixtral_queue_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler
import random
from utils import read_shared_gpt_dataset


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


def usage_example():
    model, tokenizer = initialize_model_and_tokenizer()
    
    prompts = read_shared_gpt_dataset("./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 1024)
    
    # Calculate and print lengths
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    # for i, length in enumerate(prompt_lengths):
    #     print(f"Prompt {i} length: {length}")
    avg_length = sum(prompt_lengths) / len(prompt_lengths)
    print(f"Number of prompts: {len(prompts)}")
    print(f"\nAverage prompt length: {avg_length:.2f} tokens")
    # Print quartiles
    prompt_lengths.sort()
    q1 = prompt_lengths[len(prompt_lengths) // 4]
    q2 = prompt_lengths[len(prompt_lengths) // 2]
    q3 = prompt_lengths[3 * len(prompt_lengths) // 4]
    print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}\n")
    
    
    scheduler = Scheduler(model, tokenizer)

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()