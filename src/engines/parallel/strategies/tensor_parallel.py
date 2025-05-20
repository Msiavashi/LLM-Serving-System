import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module
from .base_strategy import BaseParallelismStrategy

class TensorParallelismStrategy(BaseParallelismStrategy):
    """
    1D tensor parallelism strategy using torch.distributed.
    """

    def __init__(self, dist_manager=None, group=None):
        super().__init__(dist_manager=dist_manager)
        self.world_size = dist_manager.world_size if dist_manager else dist.get_world_size()
        self.rank = dist_manager.rank if dist_manager else dist.get_rank()
        self.group = group if group is not None else dist.group.WORLD
        self.device = dist_manager.device if dist_manager else torch.device("cpu")
        self._tensor_shape = None
        self._scaling_factor = 1
        self.device_mesh = self._create_device_mesh()

    def _create_device_mesh(self):
        mesh_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = init_device_mesh(mesh_device_type, (self.world_size,))
        return mesh

    def set_scaling_factor(self, factor: int) -> None:
        """Set scaling factor to adjust parallelism at runtime"""
        self._scaling_factor = max(1, factor)
    
    def get_effective_world_size(self) -> int:
        """Get effective world size considering the scaling factor"""
        return max(1, self.world_size // self._scaling_factor)
    
    def split_batch(self, batch):
        """
        Splits the batch along the first dimension for tensor parallelism.
        Returns the local chunk for this rank.
        """
        effective_world_size = self.get_effective_world_size()
        effective_rank = self.rank % effective_world_size
        
        # Assume batch is a tensor or a tuple/list of tensors
        if isinstance(batch, torch.Tensor):
            chunks = torch.chunk(batch, effective_world_size, dim=0)
            return chunks[effective_rank]
        elif isinstance(batch, (tuple, list)):
            # Split each tensor in the batch
            return [torch.chunk(t, effective_world_size, dim=0)[effective_rank] for t in batch]
        elif isinstance(batch, dict):
            # Support dictionary of tensors
            return {k: torch.chunk(v, effective_world_size, dim=0)[effective_rank] 
                   if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        else:
            raise ValueError("Unsupported batch type for tensor parallelism.")

    def gather_outputs(self, local_output):
        """
        Gathers local outputs from all ranks and concatenates along the first dimension.
        Supports various output types including tensors, lists, and dictionaries.
        """
        effective_world_size = self.get_effective_world_size()
        
        if isinstance(local_output, torch.Tensor):
            outputs = [torch.zeros_like(local_output) for _ in range(effective_world_size)]
            dist.all_gather(outputs, local_output, group=self.group)
            return torch.cat(outputs, dim=0)
        elif isinstance(local_output, list) and all(isinstance(item, torch.Tensor) for item in local_output):
            # For list of tensors (e.g., from batched outputs)
            gathered = []
            for item in local_output:
                item_outputs = [torch.zeros_like(item) for _ in range(effective_world_size)]
                dist.all_gather(item_outputs, item, group=self.group)
                gathered.append(torch.cat(item_outputs, dim=0))
            return gathered
        elif isinstance(local_output, dict):
            # For dictionary outputs
            result = {}
            for k, v in local_output.items():
                if isinstance(v, torch.Tensor):
                    outputs = [torch.zeros_like(v) for _ in range(effective_world_size)]
                    dist.all_gather(outputs, v, group=self.group)
                    result[k] = torch.cat(outputs, dim=0)
                else:
                    result[k] = v  # Non-tensor values are kept as is
            return result
        else:
            raise ValueError(f"Unsupported output type for gather: {type(local_output)}")

    @property
    def tensor_shape(self):
        """Get the tensor shape for communication operations"""
        if self._tensor_shape is None:
            raise ValueError("Tensor shape not set. Call set_tensor_shape() first.")
        return self._tensor_shape
    
    def set_tensor_shape(self, shape):
        """Set the tensor shape for communication operations"""
        self._tensor_shape = shape
        return self
    
    def update_from_sample(self, sample_tensor):
        """Update tensor shape based on a sample tensor"""
        if isinstance(sample_tensor, torch.Tensor):
            self._tensor_shape = sample_tensor.shape
        return self

    def parallelize_model(self, model):
        if not self.parallelize_plan:
            raise ValueError("parallelize_plan must be set before calling parallelize_model.")
        # If the plan is a function, call it with the model to get the dict
        plan = self.parallelize_plan(model) if callable(self.parallelize_plan) else self.parallelize_plan
        return parallelize_module(
            model,
            self.device_mesh,
            plan
        )

    def __repr__(self):
        return (f"<TensorParallelismStrategy rank={self.rank} world_size={self.world_size} "
                f"scaling_factor={self._scaling_factor}>")
