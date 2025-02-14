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
