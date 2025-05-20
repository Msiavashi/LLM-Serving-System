from typing import Tuple, List, Type, Optional, Dict, Any
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from src.schedulers.round_robin_scheduler import ModelInstance
from src.models.mixtral_model import MyCustomMixtral as MixtralModel
from src.models.mixtral_queue_model import MyCustomMixtral as MixtralQueueModel
from src.models.phimoe_model import PhiMoe as PhiMoeModel
from src.models.phimoe_queue_model import PhiMoe as PhiMoeQueueModel
from src.models.llama_8b import Llama8B as Llama8BModel
# Import TensorParallelismStrategy to check its type
from src.engines.parallel.strategies.tensor_parallel import TensorParallelismStrategy

class ModelFactory:
    MIXTRAL_CHECKPOINT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    PHIMOE_CHECKPOINT = "microsoft/Phi-3.5-MoE-instruct"
    LLAMA3_8B_CHECKPOINT = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    @staticmethod
    def _init_model(model_class: Type[PreTrainedModel], checkpoint: str, 
                    dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """
        Enhanced model initialization with distributed and parallel support
        """
        config = AutoConfig.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Set tokenizer padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            "config": config,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16, # Keep a default dtype
        }

        # Conditionally apply quantization_config
        # Disable quantization if TensorParallelismStrategy is used, as it might conflict
        if not isinstance(strategy, TensorParallelismStrategy):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {'': str(dist_manager.device)} # device_map often used with quantization
        # else:
        #     # For TP, device_map might be handled by parallelize_module or needs to be 'auto' or specific per rank
        #     # If parallelize_module handles device placement, explicit device_map here might conflict.
        #     # For simplicity, let's assume parallelize_module will place shards correctly.
        #     # If not using quantization, ensure model is loaded to the correct device for the rank.
        #     model_kwargs["device_map"] = {'': str(dist_manager.device)}


        # Initialize the model
        model = model_class.from_pretrained(
            checkpoint,
            **model_kwargs
        )
        
        # Parallelize model if strategy supports it
        if strategy and hasattr(strategy, "parallelize_model"):
            model = strategy.parallelize_model(model)
        
        # Create model instance
        device = dist_manager.device if not isinstance(strategy, TensorParallelismStrategy) else strategy.device_mesh.device_type
        
        # If model is already a DTensor due to parallelize_module, its .device might be tricky.
        # For ModelInstance, we need a representative device string.
        # If parallelize_module was used, the model is now potentially sharded across devices.
        # The concept of a single device for the ModelInstance might need refinement for TP.
        # Using the dist_manager's rank-specific device for now.
        model_instance_device = str(dist_manager.device)

        model_instance = ModelInstance(model, model_instance_device)
        
        return model_instance, tokenizer

    @staticmethod
    def create_model(model_type: str, dist_manager=None, strategy=None, **kwargs):
        """
        Create models with support for distributed training and inference
        """
        if model_type == "mixtral":
            return ModelFactory.create_mixtral_model(dist_manager=dist_manager, strategy=strategy, **kwargs)
        elif model_type == "mixtral_queue":
            return ModelFactory.create_mixtral_queue_model(dist_manager=dist_manager, strategy=strategy, **kwargs)
        elif model_type == "phimoe":
            return ModelFactory.create_phimoe_model(dist_manager=dist_manager, strategy=strategy, **kwargs)
        elif model_type == "phimoe_queue":
            return ModelFactory.create_phimoe_queue_model(dist_manager=dist_manager, strategy=strategy, **kwargs)
        elif model_type == "llama3_8b":
            return ModelFactory.create_llama3_8b_model(dist_manager=dist_manager, strategy=strategy, **kwargs)
        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_mixtral_model(dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """Initialize Mixtral model and tokenizer for given rank with distributed support"""
        return ModelFactory._init_model(MixtralModel, ModelFactory.MIXTRAL_CHECKPOINT, dist_manager, strategy=strategy)

    @staticmethod
    def create_mixtral_queue_model(dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """Initialize Mixtral Queue model and tokenizer for given rank with distributed support"""
        return ModelFactory._init_model(MixtralQueueModel, ModelFactory.MIXTRAL_CHECKPOINT, dist_manager, strategy=strategy)

    @staticmethod
    def create_phimoe_model(dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """Initialize Phi MOE model and tokenizer for given rank with distributed support"""
        return ModelFactory._init_model(PhiMoeModel, ModelFactory.PHIMOE_CHECKPOINT, dist_manager, strategy=strategy)

    @staticmethod
    def create_phimoe_queue_model(dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """Initialize Phi MOE Queue model and tokenizer for given rank with distributed support"""
        return ModelFactory._init_model(PhiMoeQueueModel, ModelFactory.PHIMOE_CHECKPOINT, dist_manager, strategy=strategy)

    @staticmethod
    def create_llama3_8b_model(dist_manager=None, strategy=None) -> Tuple[Any, Any]:
        """Initialize Llama 3 8B model and tokenizer for given rank with distributed support"""
        return ModelFactory._init_model(Llama8BModel, ModelFactory.LLAMA3_8B_CHECKPOINT, dist_manager, strategy=strategy)