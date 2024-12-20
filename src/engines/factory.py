from typing import Dict, Type
from .base_engine import BaseEngine
from .model_engine import ModelEngine
from .mpi_engine import MPIEngine

class EngineFactory:
    _engines: Dict[str, Type[BaseEngine]] = {
        "model": ModelEngine,  # Single model execution
        "mpi": MPIEngine,     # Distributed execution
    }

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[BaseEngine]) -> None:
        cls._engines[name] = engine_class

    @classmethod
    def create_engine(cls, name: str, model=None, **kwargs) -> BaseEngine:
        if name not in cls._engines:
            raise ValueError(f"Unknown engine type: {name}")
        
        if name == "mpi":
            return cls._engines[name](**kwargs)
        else:
            if model is None:
                raise ValueError(f"Model argument required for engine type: {name}")
            return cls._engines[name](model)
