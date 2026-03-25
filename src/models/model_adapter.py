"""ModelAdapter: Load any HuggingFace model and inject QLLM queue scheduling.

This module replaces the model-subclassing approach with composition:
- Loads models via standard AutoModelForCausalLM (no custom classes needed)
- Auto-detects MoE layers by inspecting module attributes
- Wraps MoE blocks with QueueAwareMoEWrapper at runtime
- Enables all HF-native optimizations (Flash Attention, SDPA, torch.compile, etc.)
"""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# Well-known MoE block attribute names across HF models
_MOE_ATTR_NAMES = [
    "block_sparse_moe",     # Mixtral
    "sparse_moe",           # Some variants
    "moe",                  # Generic
    "mlp",                  # Check if it has experts sub-attr
]


class ModelAdapter:
    """Loads any HuggingFace model and optionally injects per-expert queue scheduling."""

    # Well-known model checkpoints
    CHECKPOINTS = {
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mixtral_queue": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "phimoe": "microsoft/Phi-3.5-MoE-instruct",
        "phimoe_queue": "microsoft/Phi-3.5-MoE-instruct",
        "llama3_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }

    def __init__(
        self,
        checkpoint: str,
        rank: int = 0,
        attn_implementation: str = "flash_attention_2",
        load_in_4bit: bool = True,
        torch_dtype=torch.float16,
        compile_model: bool = False,
    ):
        self.checkpoint = checkpoint
        self.rank = rank
        self.device = f"cuda:{rank}"
        self.moe_wrappers: List[nn.Module] = []
        self.is_moe = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        # Resolve attention implementation with fallback
        attn_impl = self._resolve_attn_implementation(attn_implementation)

        # Load model via standard HF API — no subclassing
        logger.info(f"Loading {checkpoint} with attn={attn_impl}, 4bit={load_in_4bit}")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map={"": self.device},
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        # Auto-detect MoE architecture
        self._moe_block_names = list(self._detect_moe_blocks())
        self.is_moe = len(self._moe_block_names) > 0
        if self.is_moe:
            config = self.model.config
            n_experts = getattr(config, "num_local_experts", "?")
            top_k = getattr(config, "num_experts_per_tok", "?")
            logger.info(
                f"Detected MoE model: {len(self._moe_block_names)} layers, "
                f"{n_experts} experts, top-{top_k}"
            )
        else:
            logger.info("Dense model detected (no MoE layers)")

        # Optional torch.compile
        if compile_model:
            logger.info("Compiling model with torch.compile (reduce-overhead mode)")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _resolve_attn_implementation(self, preferred: str) -> str:
        """Try preferred attention, fall back to sdpa then eager."""
        for impl in [preferred, "sdpa", "eager"]:
            try:
                # Quick validation — some implementations need specific hardware
                if impl == "flash_attention_2":
                    import flash_attn  # noqa: F401
                return impl
            except (ImportError, Exception):
                logger.warning(f"Attention implementation '{impl}' not available, trying next")
                continue
        return "eager"

    def _detect_moe_blocks(self):
        """Auto-detect MoE layers by looking for modules with 'experts' + 'gate' attributes."""
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "gate"):
                if isinstance(module.experts, nn.ModuleList) and len(module.experts) > 1:
                    yield name, module

    def inject_queues(self, queue_manager=None):
        """Replace MoE blocks with queue-aware wrappers for per-expert scheduling."""
        from src.mixins.queue_aware_moe_wrapper import QueueAwareMoEWrapper

        self.moe_wrappers = []
        for name, moe_block in self._moe_block_names:
            wrapper = QueueAwareMoEWrapper(moe_block, queue_manager)
            # Navigate to parent module and replace attribute
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapper)
            self.moe_wrappers.append(wrapper)
            logger.info(f"Injected queue wrapper at {name}")

        logger.info(f"Injected {len(self.moe_wrappers)} queue-aware MoE wrappers")

    def set_decode_mode(self, decode: bool):
        """Switch all MoE wrappers between prefill (pass-through) and decode (queued) mode."""
        for wrapper in self.moe_wrappers:
            wrapper.set_decode_mode(decode)

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        """Standard HF forward pass — uses native attention, cache, etc."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @classmethod
    def from_name(cls, model_name: str, rank: int = 0, **kwargs) -> "ModelAdapter":
        """Create a ModelAdapter from a well-known model name."""
        checkpoint = cls.CHECKPOINTS.get(model_name)
        if checkpoint is None:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Available: {list(cls.CHECKPOINTS.keys())}. "
                f"Or pass a HuggingFace checkpoint path directly."
            )
        return cls(checkpoint=checkpoint, rank=rank, **kwargs)
