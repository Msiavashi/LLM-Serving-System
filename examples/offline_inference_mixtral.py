import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
# from src.models.mixtral_model import MyCustomMixtral
from src.models.mixtral_queue_model import MyCustomMixtral
from src.schedulers.scheduler import Scheduler


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
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
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
        "What are the different types of ecosystems?",
        "How does the human respiratory system work?",
        "What are the causes and effects of global warming?",
        "Describe the history and significance of the Eiffel Tower.",
        "What are the benefits of meditation?",
        "How do you create a budget plan?",
        "Explain the process of photosynthesis.",
        "What are the different types of musical instruments?",
        "How does the human circulatory system function?",
        "What is the importance of biodiversity?",
        "Describe the process of making bread.",
        "What are the key elements of a successful business plan?",
        "How do you practice mindfulness?",
        "What are the different types of art movements?",
        "Explain the process of water purification.",
        "What are the benefits of recycling?",
        "How does the human nervous system work?",
        "What is the significance of the Statue of Liberty?",
        "Describe the process of making cheese.",
        "What are the key principles of effective communication?",
        "How do you start a garden?",
        "What are the different types of computer programming languages?",
        "Explain the process of fermentation.",
        "What are the benefits of a balanced diet?",
        "How does the human skeletal system function?",
        "What is the importance of renewable energy?",
        "Describe the process of making wine.",
        "What are the key elements of a healthy lifestyle?",
        "How do you create a workout plan?",
        "What are the different types of literature genres?",
        "Explain the process of desalination.",
        "What are the benefits of regular sleep?",
        "How does the human endocrine system function?",
        "What are the causes and effects of deforestation?",
        "Describe the history and significance of the Colosseum.",
        "What are the benefits of practicing yoga?",
        "How do you create a financial plan?",
        "Explain the process of cellular respiration.",
        "What are the different types of musical genres?",
        "How does the human digestive system function?",
        "What is the importance of conservation?",
        "Describe the process of making pasta.",
        "What are the key elements of a successful marketing strategy?",
        "How do you practice gratitude?",
        "What are the different types of painting techniques?",
        "Explain the process of distillation.",
        "What are the benefits of using public transportation?",
        "How does the human reproductive system work?",
        "What is the significance of the Pyramids of Giza?",
        "Describe the process of making chocolate.",
        "What are the key principles of leadership?",
        "How do you start a small business?",
        "What are the different types of computer networks?",
        "Explain the process of osmosis.",
        "What are the benefits of a plant-based diet?",
        "How does the human muscular system function?",
        "What is the importance of sustainable development?",
        "Describe the process of making beer.",
        "What are the key elements of a successful project plan?",
        "How do you practice self-care?",
        "What are the different types of dance styles?",
        "Explain the process of evaporation.",
        "What are the benefits of regular hydration?",
        "How does the human skeletal system support the body?",
        "What is the significance of the Great Barrier Reef?",
        "Describe the process of making soap.",
        "What are the key principles of time management?",
        "How do you create a meal plan?",
        "What are the different types of poetry?",
        "Explain the process of filtration.",
        "What are the benefits of regular stretching?",
        "How does the human immune system function?",
        "What is the importance of cultural heritage?",
        "Describe the process of making candles.",
        "What are the key elements of a successful advertising campaign?",
        "How do you practice mindfulness meditation?",
        "What are the different types of sculpture techniques?",
        "Explain the process of sublimation.",
        "What are the benefits of regular exercise for mental health?",
        "How does the human cardiovascular system function?",
        "What is the significance of the Taj Mahal?",
        "Describe the process of making ice cream.",
        "What are the key principles of conflict resolution?",
        "How do you create a study plan?",
        "What are the different types of drama?",
        "Explain the process of crystallization.",
        "What are the benefits of regular social interaction?",
        "How does the human respiratory system support the body?",
        "What is the importance of biodiversity conservation?",
        "Describe the process of making yogurt.",
        "What are the key elements of a successful negotiation?",
        "How do you practice active listening?",
        "What are the different types of ceramics?",
        "Explain the process of condensation.",
        "What are the benefits of regular outdoor activities?",
        "How does the human nervous system control the body?",
        "What is the significance of the Leaning Tower of Pisa?",
        "Describe the process of making butter.",
        "What are the key principles of effective teamwork?",
        "How do you create a travel itinerary?",
        "What are the different types of fiction genres?",
        "Explain the process of precipitation.",
        "What are the benefits of regular mental exercises?",
        "How does the human excretory system function?",
        "What is the importance of environmental sustainability?",
        "Describe the process of making cheese.",
        "What are the key elements of a successful event plan?",
        "How do you practice positive thinking?",
        "What are the different types of printmaking techniques?",
        "Explain the process of transpiration.",
        "What are the benefits of regular creative activities?",
        "How does the human lymphatic system function?",
        "What is the significance of the Statue of Liberty?",
        "Describe the process of making jam.",
        "What are the key principles of effective communication?",
        "How do you create a fitness routine?",
        "What are the different types of non-fiction genres?",
        "Explain the process of photosynthesis.",
        "What are the benefits of regular volunteering?",
        "How does the human sensory system function?",
        "What is the importance of historical preservation?",
        "Describe the process of making pickles.",
        "What are the key elements of a successful business strategy?",
        "How do you practice emotional intelligence?",
        "What are the different types of textile techniques?",
        "Explain the process of fermentation.",
        "What are the benefits of regular journaling?",
        "How does the human integumentary system function?",
        "What is the significance of the Eiffel Tower?",
        "Describe the process of making bread.",
        "What are the key principles of effective leadership?",
        "How do you create a budget?",
        "What are the different types of visual art?",
        "Explain the process of distillation.",
        "What are the benefits of regular reading?",
        "How does the human skeletal system support the body?",
        "What is the importance of renewable energy?",
        "Describe the process of making wine.",
        "What are the key elements of a healthy lifestyle?",
        "How do you create a workout plan?",
        "What are the different types of literature genres?",
        "Explain the process of desalination.",
        "What are the benefits of regular sleep?",
        "Write a detailed essay on the impact of climate change on global agriculture, including the effects on crop yields, soil health, water availability, and the socio-economic implications for farmers and communities around the world. Discuss potential mitigation strategies and the role of technology in adapting to these changes.",
    ]
    
    # random.shuffle(prompts)
    
    scheduler = Scheduler(model, tokenizer)

    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    results = scheduler.run_scheduler()

    # for seq in results:
    #     generated_text = seq.get_generated_text(tokenizer)
    #     print(f"Prompt: {seq.prompt}\nGenerated Text: {generated_text}\n")


if __name__ == "__main__":
    usage_example()