import sys
import os
import asyncio
import numpy as np
from time import time
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import ConfigManager
from utils import read_shared_gpt_dataset

async def process_prompt(client, prompt, model_id, priority=0):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
                "priority": priority  # Move priority to message level
            }],
            model=model_id,
            max_tokens=50,
            temperature=0.7
        )
        
        # result = chat_completion.choices[0].message.content
        # print(f"Prompt: {prompt}\nGenerated Text: {result}\n")
        
    except Exception as e:
        print(f"Error processing prompt: {str(e)}")

async def main():
    # Get configuration
    config = ConfigManager()
    api_base = f"http://{config.get('server.host', 'localhost')}:{config.get('server.port', 8000)}/v1"
    model_id = config.get('model.checkpoint', 'mistralai/Mixtral-8x7B-Instruct-v0.1')

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base
    )

    # Read dataset
    prompts = read_shared_gpt_dataset("./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 1024)
    
    # Generate Poisson arrival times (10 requests per second)
    arrival_rate = 5  # requests per second
    num_requests = len(prompts)
    intervals = np.random.exponential(1.0/arrival_rate, num_requests)
    arrival_times = np.cumsum(intervals)
    
    # Calculate number of latency-sensitive requests (25%)
    num_latency_sensitive = int(len(prompts) * 0.2)
    
    # Generate exponential distribution scores
    exp_scores = np.random.exponential(scale=1.0, size=len(prompts))
    
    # Sort indices by exponential scores and select top 30% as high priority
    high_priority_indices = np.argsort(exp_scores)[-num_latency_sensitive:]
    
    # Process requests with Poisson timing
    start_time = time()
    tasks = []
    
    for i, (prompt, arrival_time) in enumerate(zip(prompts, arrival_times)):
        # Wait until the next arrival time
        await asyncio.sleep(max(0, arrival_time - (time() - start_time)))
        # Set priority based on index
        priority = 1 if i in high_priority_indices else 0
        # Create and start task
        task = asyncio.create_task(process_prompt(client, prompt, model_id, priority))
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
