import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from src.schedulers.scheduler import Scheduler
# from src.models.phimoe_model import PhiMoe
from src.models.phimoe_queue_model import PhiMoe

def initialize_model_and_tokenizer():
    config = AutoConfig.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = PhiMoe.from_pretrained(
        "microsoft/Phi-3.5-MoE-instruct",
        config=config,
        device_map='auto',
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer

# Rest of the code remains the same
def usage_example():
    model, tokenizer = initialize_model_and_tokenizer()
    
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
    
    scheduler = Scheduler(model, tokenizer)

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()
