from typing import Dict, Type
from .base_engine import BaseEngine
from .model_engine import ModelEngine
from .async_model_engine import AsyncModelEngine
from .distributed_engine import DistributedEngine

class EngineFactory:
    _engines: Dict[str, Type[BaseEngine]] = {
        "model": ModelEngine,  # Single model execution
        "async_model": AsyncModelEngine,
        "distributed": DistributedEngine  # Add distributed engine
    }

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[BaseEngine]) -> None:
        cls._engines[name] = engine_class

    @classmethod
    def create_engine(cls, name: str, model=None, **kwargs) -> BaseEngine:
        if name not in cls._engines:
            if name == "mpi":
                # Lazy import MPIEngine only when needed
                from .mpi_engine import MPIEngine
                cls._engines["mpi"] = MPIEngine
            else:
                raise ValueError(f"Unknown engine type: {name}")
        
        if name == "mpi":
            return cls._engines[name](**kwargs)
        elif name == "distributed":
            if "strategy" not in kwargs or "dist_manager" not in kwargs:
                raise ValueError("Both 'strategy' and 'dist_manager' are required for distributed engine")
            return cls._engines[name](**kwargs)
        else:
            if model is None:
                raise ValueError(f"Model argument required for engine type: {name}")
            return cls._engines[name](model)
