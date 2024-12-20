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
from src.models.model_factory import ModelFactory
from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory

def usage_example():
    
    # Create model using factory - without specifying rank for single GPU usage
    model_instances, tokenizer = ModelFactory.create_mixtral_model(rank=1)
    model = model_instances[0].model
    
    # Create engine using factory
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=32
    )
    
    prompts = read_shared_gpt_dataset("./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 1024)
    
    # Calculate and print lengths
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    avg_length = sum(prompt_lengths) / len(prompt_lengths)
    print(f"Number of prompts: {len(prompts)}")
    print(f"\nAverage prompt length: {avg_length:.2f} tokens")
    # Print quartiles
    prompt_lengths.sort()
    q1 = prompt_lengths[len(prompt_lengths) // 4]
    q2 = prompt_lengths[len(prompt_lengths) // 2]
    q3 = prompt_lengths[3 * len(prompt_lengths) // 4]
    print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}\n")
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")

if __name__ == "__main__":
    usage_example()