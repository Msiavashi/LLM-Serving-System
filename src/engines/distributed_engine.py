import torch
import torch.distributed as dist
from typing import Any, List
from .base_engine import BaseEngine

class DistributedEngine(BaseEngine):
    def __init__(self, dist_manager, strategy):
        self.dist_manager = dist_manager
        self.strategy = strategy
 
    def run_batch(self, batch) -> Any:
        self.dist_manager.barrier()
        outputs = self.strategy.forward(batch)
        return outputs