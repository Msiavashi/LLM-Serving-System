from typing import Tuple, List, Type
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from src.schedulers.round_robin_scheduler import ModelInstance
from src.models.mixtral_model import MyCustomMixtral as MixtralModel
from src.models.mixtral_queue_model import MyCustomMixtral as MixtralQueueModel
from src.models.phimoe_model import PhiMoe as PhiMoeModel
from src.models.phimoe_queue_model import PhiMoe as PhiMoeQueueModel
from src.models.llama_8b import Llama8B as Llama8BModel

class ModelFactory:
    MIXTRAL_CHECKPOINT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    PHIMOE_CHECKPOINT = "microsoft/Phi-3.5-MoE-instruct"
    LLAMA3_8B_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"

    @staticmethod
    def _init_model(model_class: Type[PreTrainedModel], rank: int, checkpoint: str) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Common initialization logic for Mixtral models"""
        config = AutoConfig.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        
        # Enable flash attention with variable length support
        # config.use_flash_attention_2 = True 
        # config.use_flash_attention_varlen = True
        
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
            checkpoint,
            config=config,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            # Add attn_implementation for flash attention
            # attn_implementation="flash_attention_2",
            # attn_implementation="eager",
        )
        
        return [ModelInstance(model, f"cuda:{rank}")], tokenizer

    @staticmethod
    def create_model(model_type: str, rank: int = 1, **kwargs):
        if model_type == "mixtral":
            return ModelFactory.create_mixtral_model(rank=rank, **kwargs)
        elif model_type == "mixtral_queue":
            return ModelFactory.create_mixtral_queue_model(rank=rank, **kwargs)
        elif model_type == "phimoe":
            return ModelFactory.create_phimoe_model(rank=rank, **kwargs)
        elif model_type == "phimoe_queue":
            return ModelFactory.create_phimoe_queue_model(rank=rank, **kwargs)
        elif model_type == "llama3_8b":
            return ModelFactory.create_llama3_8b_model(rank=rank, **kwargs)
        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_mixtral_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Mixtral model and tokenizer for given rank"""
        return ModelFactory._init_model(MixtralModel, rank, ModelFactory.MIXTRAL_CHECKPOINT)

    @staticmethod
    def create_mixtral_queue_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Mixtral Queue model and tokenizer for given rank"""
        return ModelFactory._init_model(MixtralQueueModel, rank, ModelFactory.MIXTRAL_CHECKPOINT)

    @staticmethod
    def create_phimoe_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Phi MOE model and tokenizer for given rank"""
        return ModelFactory._init_model(PhiMoeModel, rank, ModelFactory.PHIMOE_CHECKPOINT)

    @staticmethod
    def create_phimoe_queue_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Phi MOE Queue model and tokenizer for given rank"""
        return ModelFactory._init_model(PhiMoeQueueModel, rank, ModelFactory.PHIMOE_CHECKPOINT)

    @staticmethod
    def create_llama3_8b_model(rank: int) -> Tuple[List[ModelInstance], AutoTokenizer]:
        """Initialize Llama 3 8B model and tokenizer for given rank"""
        return ModelFactory._init_model(Llama8BModel, rank, ModelFactory.LLAMA3_8B_CHECKPOINT)