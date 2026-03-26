from src.engines.base_engine import BaseEngine
from src.engines.qllm_engine import QllmEngine


class EngineFactory:

    @classmethod
    def create_engine(cls, name: str = "qllm", **kwargs) -> BaseEngine:
        if name == "qllm":
            return QllmEngine(**kwargs)
        raise ValueError(f"Unknown engine type: {name}. Available: 'qllm'")
