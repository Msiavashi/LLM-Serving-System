from typing import Tuple, Type, Optional, Dict, Any
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from src.schedulers.round_robin_scheduler import ModelInstance
# from src.models.mixtral_model import MyCustomMixtral as MixtralModel
# from src.models.mixtral_queue_model import MyCustomMixtral as MixtralQueueModel
# from src.models.phimoe_model import PhiMoe as PhiMoeModel
# from src.models.phimoe_queue_model import PhiMoe as PhiMoeQueueModel
from src.models.llama_8b import Llama8B as Llama8BModel

class ModelFactory:
    MODEL_CLASSES = {
        # "mixtral": (MixtralModel, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        # "mixtral_queue": (MixtralQueueModel, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        # "phimoe": (PhiMoeModel, "microsoft/Phi-3.5-MoE-instruct"),
        # "phimoe_queue": (PhiMoeQueueModel, "microsoft/Phi-3.5-MoE-instruct"),
        "llama3_8b": (Llama8BModel, "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    }

    @staticmethod
    def get_model_class(model_type: str) -> Optional[Type[PreTrainedModel]]:
        return ModelFactory.MODEL_CLASSES.get(model_type, (None,))[0]

    @staticmethod
    def get_tokenizer_for(model_name: str):
        _, checkpoint = ModelFactory.MODEL_CLASSES[model_name]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def _init_model(model_class: Type[PreTrainedModel], checkpoint: str, device: str, **kwargs) -> Tuple[Any, Any]:
        config = AutoConfig.from_pretrained(checkpoint)
        model_kwargs = {
            "config": config,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            "device_map": device,
        }
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        )
        # model_kwargs["quantization_config"] = quantization_config
        model_kwargs["quantization_config"] = None
        model = model_class.from_pretrained(checkpoint, **model_kwargs)
        return ModelInstance(model=model, device=device)

    @staticmethod
    def create_model(model_type: str, device: str="cuda:0", **kwargs):
        if model_type not in ModelFactory.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        model_class, checkpoint = ModelFactory.MODEL_CLASSES[model_type]
        return ModelFactory._init_model(model_class, checkpoint, device, **kwargs)