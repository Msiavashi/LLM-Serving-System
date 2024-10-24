
from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from mixtral_model import MyCustomMixtral
from scheduler import Scheduler
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
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = MyCustomMixtral.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        config=config,
        device_map='auto',
        quantization_config=quantization_config,
    )
    
    return model, tokenizer


def usage_example():
    model, tokenizer = initialize_model_and_tokenizer()
    prompts = [
        "What is the meaning of life?", 
        "What is the capital of France?", 
        "What is the largest mammal?", 
        "What is the most popular programming language?", 
        "What is the best movie of all time?"
    ]
    
    random.shuffle(prompts)
    
    scheduler = Scheduler(model, tokenizer)

    # Add sequences to the queue
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    # Run scheduler
    results = scheduler.run_scheduler()

    # Retrieve and print results
    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()