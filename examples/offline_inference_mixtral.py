import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from src.engines.factory import EngineFactory
from utils import generate_prompts


def usage_example():
    np.random.seed(42)  # Set the random seed for reproducibility
    # Create model using factory
    # model_instances, tokenizer = ModelFactory.create_mixtral_queue_model(rank=1)
    model_instances, tokenizer = ModelFactory.create_mixtral_model(rank=1)
    model = model_instances[0].model
    
    # Create engine using factory - now using standard model engine 
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory with engine instead of model
    scheduler = SchedulerFactory.create_scheduler(
        name="priority",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=32
    )
    
    prompts = generate_prompts(256, 16, tokenizer)
    
    # Calculate number of latency-sensitive requests (2%)
    num_latency_sensitive = int(len(prompts) * 0.3)  # 2% of total prompts
    
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