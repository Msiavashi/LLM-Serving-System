import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from utils import generate_prompts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def usage_example():
    model_instances, tokenizer = ModelFactory.create_model("llama3_8b", rank=0)
    model = model_instances[0].model
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
    
    prompts = generate_prompts(num_prompts=64, prompt_size=512, tokenizer=tokenizer)
    
    
    # Create engine using factory - now using standard model engine 
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory with engine instead of model
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=16
    )
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    # for seq in results:
    #     generated_text = seq.get_generated_text(tokenizer)
        
    #     # Trim padding tokens
    #     if tokenizer.pad_token:
    #         generated_text = generated_text.replace(tokenizer.pad_token, "").strip()
            
    #     print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()
