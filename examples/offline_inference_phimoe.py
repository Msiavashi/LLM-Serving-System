import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from src.engines.factory import EngineFactory
from utils import generate_prompts

def usage_example():
    # Create model using factory
    model_instances, tokenizer = ModelFactory.create_phimoe_model(rank=1)
    model = model_instances[0].model
    
    # Create engine using factory
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory with engine
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=64
    )
    
    prompts = [
        "How do I make a cake?",
        "What is the capital of France?",
        "Explain quantum computing",
        "How to tie a tie?",
        "Write a hello world program in Python",
        "What are black holes?",
        "How to make pizza dough?",
        "Explain photosynthesis",
        "What is machine learning?",
        "How do cars work?",
        "What is climate change?",
        "Explain DNA structure",
        "How to learn programming?",
        "What is artificial intelligence?",
        "How to start exercising?",
        "What is blockchain?",
        "Explain evolution theory",
        "How to write a resume?",
        "What is renewable energy?",
        "How to learn a new language?",
        "What is cryptocurrency?",
        "Explain plate tectonics",
        "How to manage time effectively?",
        "What is cloud computing?",
        "How to start meditation?",
        "What is nuclear fusion?",
        "Explain the water cycle",
        "How to improve memory?",
        "What is virtual reality?",
        "How to reduce stress?",
        "What is quantum physics?",
        "Explain the solar system",
        "How to start a business?"
    ]
    
    # Alternatively, use the generate_prompts utility like mixtral example
    # prompts = generate_prompts(128, 1024, tokenizer)
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")

if __name__ == "__main__":
    usage_example()
