import sys
import os

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
# from src.models.mixtral_model import MyCustomMixtral
from src.models.mixtral_queue_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler
import random

def read_dataset(dataset_path, num_prompts):
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Filter and Transform in a Single Step
    dataset = [(conv[0]["value"], conv[1]["value"]) 
                for data in dataset 
                if (conv := data.get("conversations", [])) and len(conv) >= 2]

    random.shuffle(dataset)
    # filter duplicates
    dataset = list(set(dataset))
    filtered_dataset = [prompt for prompt, _ in dataset][:num_prompts]
    print(f"Length of filtered dataset: {len(filtered_dataset)}")
    return filtered_dataset   

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
    )
    
    return model, tokenizer


def usage_example():
    model, tokenizer = initialize_model_and_tokenizer()
    
    prompts = read_dataset("./datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 1000)
    
    # Calculate and print lengths
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    # for i, length in enumerate(prompt_lengths):
    #     print(f"Prompt {i} length: {length}")
    avg_length = sum(prompt_lengths) / len(prompt_lengths)
    print(f"\nAverage prompt length: {avg_length:.2f} tokens")
    # Print quartiles
    prompt_lengths.sort()
    q1 = prompt_lengths[len(prompt_lengths) // 4]
    q2 = prompt_lengths[len(prompt_lengths) // 2]
    q3 = prompt_lengths[3 * len(prompt_lengths) // 4]
    print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")
    
    random.shuffle(prompts)
    
    scheduler = Scheduler(model, tokenizer)

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()