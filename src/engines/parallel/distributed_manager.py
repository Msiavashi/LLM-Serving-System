import torch
import torch.distributed as dist
import os
from typing import Dict, Any, Optional, List

class DistributedManager:
    def __init__(self, backend="nccl", init_method="env://"):
        """Initialize the distributed manager"""
        self._initialize_dist_group(backend, init_method)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = self._setup_device()
        self.process_groups = {}  # Store different process groups for scaling

    def _initialize_dist_group(self, backend, init_method):
        """Initialize the distributed process group"""
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=init_method)
    
    def _setup_device(self):
        """Setup the appropriate device for this rank"""
        if torch.cuda.is_available():
            device_id = self.rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return device
    
    def get_stage_for_rank(self, num_stages: int) -> int:
        """Map rank to pipeline stage"""
        return self.rank % num_stages
    
    def create_process_group(self, name: str, ranks: List[int]) -> Any:
        """Create a new process group with specified ranks"""
        if name in self.process_groups:
            return self.process_groups[name]
        
        group = dist.new_group(ranks=ranks)
        self.process_groups[name] = group
        return group
    
    def create_scaled_groups(self, scale_factor: int) -> Dict[str, Any]:
        """Create scaled process groups for runtime scaling"""
        groups = {}
        # Create groups for different scales
        for i in range(scale_factor):
            ranks = list(range(i, self.world_size, scale_factor))
            group_name = f"scale_{scale_factor}_{i}"
            groups[group_name] = self.create_process_group(group_name, ranks)
        return groups
    
    def barrier(self):
        """Synchronize all processes"""
        dist.barrier()
    
    def finalize(self):
        """Clean up distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    @property
    def is_master_process(self):
        """Check if this is the master process (rank 0)"""
        return self.rank == 0
    
    def __repr__(self):
        return f"<DistributedManager rank={self.rank} world_size={self.world_size} device={self.device}>"