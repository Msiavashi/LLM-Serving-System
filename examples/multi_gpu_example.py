import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from src.schedulers.round_robin_scheduler import RoundRobinScheduler, ModelInstance
from src.models.mixtral_queue_model import MyCustomMixtral


def initialize_model_and_tokenizer(num_models: int):
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
    
    models = []
    for i in range(num_models):
        device_map = {'': f'cuda:{i}'}  # Map all modules to specified GPU
        model = MyCustomMixtral.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            config=config,
            device_map=device_map,  # Use device mapping instead of moving
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        models.append(ModelInstance(model, f"cuda:{i}"))
    
    return models, tokenizer


def usage_example():
    # Initialize two model instances
    models, tokenizer = initialize_model_and_tokenizer(num_models=2)
    
    # Create sample prompts with more variety and quantity
    prompts = [
        "Tell me a story about a brave knight.",
        "Explain quantum computing in simple terms.",
        "How do airplanes stay in the air?",
        "Write a poem about spring flowers.",
        "What is machine learning and how does it work?",
        "Describe the process of photosynthesis in plants.",
        "How does the internet work?",
        "What causes earthquakes and tsunamis?",
        "What is the theory of relativity?",
        "How do black holes form?",
        "Explain the water cycle on Earth.",
        "What is artificial intelligence?",
        "How does the human brain work?",
        "What causes climate change?",
        "Describe the process of evolution.",
        "How do vaccines work?",
        "What is cryptocurrency?",
        "Explain the theory of plate tectonics.",
        "How do computers process information?",
        "What is the Big Bang theory?",
        "How does DNA replication work?",
        "What are renewable energy sources?",
        "Explain how a nuclear reactor works.",
        "What is quantum entanglement?",
        "How do stars form and die?",
        "What is machine consciousness?",
        "Explain how lightning forms.",
        "What is dark matter?",
        "How do electric cars work?",
        "What is string theory?",
        "Explain how volcanoes form.",
        "What is artificial general intelligence?"
    ] * 16  # Multiply prompts for larger batches
    
    # Create and use the round-robin scheduler with larger batch size
    scheduler = RoundRobinScheduler(models, tokenizer, batch_size=32)
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()
    
    # Print results
    for seq in results:
        print(f"\nPrompt: {seq.prompt}")
        print(f"Generated on device: {seq.device}")
        print(f"Generated text: {seq.get_generated_text(tokenizer)}")


if __name__ == "__main__":
    usage_example()
