from typing import Tuple, List
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from src.schedulers.round_robin_scheduler import ModelInstance
from src.models.mixtral_queue_model import MyCustomMixtral

class ModelFactory:
    @staticmethod
    def create_mixtral_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Mixtral model and tokenizer for given rank"""
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
        
        device_map = {'': f'cuda:{rank}'}
        model = MyCustomMixtral.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            config=config,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        return [ModelInstance(model, f"cuda:{rank}")], tokenizer
