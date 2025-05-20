import torch
import torch.distributed as dist
from typing import Any, List
from .base_engine import BaseEngine

class DistributedEngine(BaseEngine):
    def __init__(self, model_stages: List[torch.nn.Module], strategy, dist_manager):
        self.model_stages = model_stages
        self.strategy = strategy
        self.dist_manager = dist_manager
        
        # Ensure strategy has access to the distributed manager if not already set
        if hasattr(strategy, 'dist_manager') and strategy.dist_manager is None:
            strategy.dist_manager = dist_manager

    def run_batch(self, batch) -> Any:
        """Process a batch using tensor parallelism"""
        # Split the batch according to the current parallelism strategy
        local_batch = self.strategy.split_batch(batch)
        
        # Process the local batch
        local_output = self._run_microbatch(local_batch)
        
        # Gather results from all ranks
        return self.strategy.gather_outputs(local_output)

    def _run_microbatch(self, microbatch):
        input_tensor = microbatch
        for stage_idx, stage in enumerate(self.model_stages):
            if stage_idx > 0:
                input_tensor = self._recv_tensor(src=self.dist_manager.rank-1)
            
            # Update the strategy with the current tensor shape if needed
            if hasattr(self.strategy, 'update_from_sample') and isinstance(input_tensor, torch.Tensor):
                self.strategy.update_from_sample(input_tensor)
                
            input_tensor = stage(input_tensor)
            
            if stage_idx < len(self.model_stages) - 1:
                self._send_tensor(input_tensor, dst=self.dist_manager.rank+1)
        
        return input_tensor

    def _send_tensor(self, tensor, dst):
        if isinstance(tensor, torch.Tensor):
            dist.send(tensor.detach().cpu(), dst=dst)
        elif isinstance(tensor, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in tensor):
            # Send shape information first for complex structures
            shapes = [t.shape for t in tensor]
            shape_tensor = torch.tensor([len(shapes)] + [dim for shape in shapes for dim in shape], 
                                      dtype=torch.long, device='cpu')
            dist.send(shape_tensor, dst=dst)
            for t in tensor:
                dist.send(t.detach().cpu(), dst=dst)
        else:
            raise ValueError(f"Unsupported tensor type for sending: {type(tensor)}")

    def _recv_tensor(self, src):
        if hasattr(self.strategy, 'tensor_shape') and self.strategy.tensor_shape is not None:
            # Using predefined shape
            shape = self.strategy.tensor_shape
            tensor = torch.zeros(shape, dtype=torch.float32, device=self.dist_manager.device)
            dist.recv(tensor, src=src)
            return tensor
        else:
            # Receive shape information first (for dynamic tensors)
            shape_tensor = torch.zeros(64, dtype=torch.long, device='cpu')  # Assuming max 64 dimensions
            dist.recv(shape_tensor, src=src)
            
            num_tensors = shape_tensor[0].item()
            if num_tensors == 1:
                # Single tensor case
                dims = shape_tensor[1:shape_tensor[1:].nonzero()[-1] + 2].tolist()
                tensor = torch.zeros(dims, dtype=torch.float32, device=self.dist_manager.device)
                dist.recv(tensor, src=src)
                return tensor
            else:
                # Multiple tensors case
                idx = 1
                tensors = []
                for _ in range(num_tensors):
                    # Extract dimensions for this tensor
                    dim_count = shape_tensor[idx].item()
                    idx += 1
                    dims = shape_tensor[idx:idx+dim_count].tolist()
                    idx += dim_count
                    
                    # Create and receive tensor
                    tensor = torch.zeros(dims, dtype=torch.float32, device=self.dist_manager.device)
                    dist.recv(tensor, src=src)
                    tensors.append(tensor)
                
                return tensors
                
    def set_scaling_factor(self, factor: int) -> None:
        """Adjust scaling factor at runtime if strategy supports it"""
        if hasattr(self.strategy, 'set_scaling_factor'):
            self.strategy.set_scaling_factor(factor)
