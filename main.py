from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
# from src.models.mixtral_model import MyCustomMixtral
from src.models.mixtral_queue_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler
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
        bnb_4bit_compute_dtype=torch.float16,
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
        "Tell me a story about a brave knight.",
        "What are the benefits of a healthy diet?",
        "Explain the theory of relativity in simple terms.",
        "How do airplanes stay in the air?",
        "What is the capital of France?",
        "Describe the process of photosynthesis.",
        "What are the main causes of climate change?",
        "How does blockchain technology work?",
        "What are the symptoms of the common cold?",
        "Explain the concept of artificial intelligence.",
        "What is the history of the internet?",
        "How do you make a perfect cup of coffee?",
        "What are the different types of renewable energy?",
        "Describe the life cycle of a butterfly.",
        "What are the key principles of democracy?",
        "How do you play the game of chess?",
        "What is the significance of the Great Wall of China?",
        "Explain the process of human digestion.",
        "What are the benefits of regular exercise?",
        "How does the stock market work?",
        "What is the importance of mental health?",
        "Describe the structure of the human brain.",
        "What are the different types of clouds?",
        "How do you bake a chocolate cake?",
        "What is the role of the United Nations?",
        "Explain the concept of quantum computing.",
        "What are the main functions of the human liver?",
        "How do you grow a vegetable garden?",
        "What is the history of the Roman Empire?",
        "Describe the process of cell division.",
        "What are the benefits of learning a second language?",
        "How does the immune system protect the body?"
    ] * 4
    
    random.shuffle(prompts)
    
    scheduler = Scheduler(model, tokenizer)

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    for seq in results:
        generated_text = seq.get_generated_text(tokenizer)
        print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()