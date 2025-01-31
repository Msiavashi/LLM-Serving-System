import sys
import os

import numpy as np

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_shared_gpt_dataset
from src.models.model_factory import ModelFactory
from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory

def usage_example():
    np.random.seed(42)  # Set the random seed for reproducibility
    
    # Create model using factory - without specifying rank for single GPU usage
    model_instances, tokenizer = ModelFactory.create_mixtral_queue_model(rank=1)
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
    
    # Calculate number of latency-sensitive requests (2%)
    num_latency_sensitive = int(len(prompts) * 0.05)  # 2% of total prompts
    
    # Generate exponential distribution scores
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    
    # Sort indices by exponential scores and select top 2% as high priority
    high_priority_indices = np.argsort(exp_scores)[-num_latency_sensitive:]
    for i, prompt in enumerate(prompts):
        priority = 1 if i in high_priority_indices else 0
        scheduler.add_sequence_to_queue(prompt, priority=priority)
    
    results = scheduler.run_scheduler()

    # for seq in results:
    #     generated_text = seq.get_generated_text(tokenizer)
    #     print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")

if __name__ == "__main__":
    usage_example()