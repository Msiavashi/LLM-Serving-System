import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from src.engines.factory import EngineFactory
from utils import generate_prompts


def usage_example():
    np.random.seed(42)  # Set the random seed for reproducibility
    # Create model using factory
    model_instances, tokenizer = ModelFactory.create_mixtral_model(rank=1)
    # model_instances, tokenizer = ModelFactory.create_mixtral_queue_model(rank=0)
    model = model_instances[0].model
    
    # Create engine using factory - now using standard model engine 
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory with engine instead of model
    scheduler = SchedulerFactory.create_scheduler(
        name="priority",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=4
    )
    
    # Set the scheduler into the model through the new setter method.
    # Note: this is not of all models. Only models supporting per-expert queues should set this.
    model.set_scheduler(scheduler)
    
    # prompts = generate_prompts(256, 16, tokenizer)
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "Describe the process of photosynthesis in detail, including the role of chlorophyll and sunlight.",
        "Summarize the plot of 'Pride and Prejudice' in one sentence.",
        # generate 4 more prompts
        "What are the main differences between classical and quantum computing?",
        "How does the human brain process information?",
        "What are the key principles of effective time management?",
        "Explain the significance of the Turing test in artificial intelligence.",
        "Describe the impact of climate change on global ecosystems.",
        # Add one significantly longer prompt
        "Discuss the ethical implications of genetic engineering in humans, including potential benefits and risks."
    ]
    
    # Calculate number of latency-sensitive requests (20%)
    num_latency_sensitive = int(len(prompts) * 0.2)
    
    # Generate exponential distribution scores
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    
    # Sort indices by exponential scores and select top 20% as high priority
    high_priority_indices = np.argsort(exp_scores)[-num_latency_sensitive:]
    
    for i, prompt in enumerate(prompts):
        priority = 1 if i in high_priority_indices else 0
        scheduler.add_sequence_to_queue(prompt, priority=priority)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()
