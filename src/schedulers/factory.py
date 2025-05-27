from typing import Dict, Type
from .base_scheduler import BaseScheduler
from .fcfs_scheduler import FCFSScheduler
from .round_robin_scheduler import RoundRobinScheduler
from .async_fcfs_scheduler import AsyncFCFSScheduler
from .priority_scheduler import PriorityScheduler
from .async_priority_scheduler import AsyncPriorityScheduler

class SchedulerFactory:
    _schedulers: Dict[str, Type[BaseScheduler]] = {
        "fcfs": FCFSScheduler,
        "round_robin": RoundRobinScheduler,
        "async_fcfs": AsyncFCFSScheduler,
        "priority": PriorityScheduler,
        "async_priority": AsyncPriorityScheduler
    }

    @classmethod
    def register_scheduler(cls, name: str, scheduler_class: Type[BaseScheduler]) -> None:
        cls._schedulers[name] = scheduler_class

    @classmethod
    def create_scheduler(cls, name: str, engine, tokenizer, **kwargs) -> BaseScheduler:
        if name not in cls._schedulers:
            raise ValueError(f"Unknown scheduler type: {name}")
        
        return cls._schedulers[name](engine=engine, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def create_scheduler(name, engine, tokenizer, batch_size=32, **kwargs):
        """
        Creates a scheduler based on the specified type.
        
        Args:
            name: The type of scheduler to create
            engine: The engine to use for execution
            tokenizer: The tokenizer for processing input/output
            batch_size: The batch size for processing
            **kwargs: Additional arguments for specific scheduler types
            
        Returns:
            An instance of the requested scheduler
        """
        if name.lower() == "fcfs":
            return FCFSScheduler(engine, tokenizer, batch_size)
        # Add more scheduler types as needed
        else:
            raise ValueError(f"Unknown scheduler type: {name}")
