from typing import Tuple, List, Type
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from src.schedulers.round_robin_scheduler import ModelInstance
from src.models.mixtral_model import MyCustomMixtral as MixtralModel
from src.models.mixtral_queue_model import MyCustomMixtral as MixtralQueueModel

class ModelFactory:
    @staticmethod
    def _init_model(model_class: Type[PreTrainedModel], rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Common initialization logic for Mixtral models"""
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
        model = model_class.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            config=config,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        return [ModelInstance(model, f"cuda:{rank}")], tokenizer

    @staticmethod
    def create_mixtral_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Mixtral model and tokenizer for given rank"""
        return ModelFactory._init_model(MixtralModel, rank)

    @staticmethod
    def create_mixtral_queue_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Mixtral Queue model and tokenizer for given rank"""
        return ModelFactory._init_model(MixtralQueueModel, rank)
