import torch
from typing import Any, List

class PipelineParallelStrategy:
    def __init__(self, num_stages: int, microbatch_size: int, tensor_shape: tuple):
        self.num_stages = num_stages
        self.microbatch_size = microbatch_size
        self.tensor_shape = tensor_shape

    def split_model(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        # User must implement model splitting logic
        # Example: return [model.part1, model.part2, ...]
        raise NotImplementedError

    def split_batch(self, batch: Any) -> List[Any]:
        # Split batch into microbatches
        return [batch[i:i+self.microbatch_size] for i in range(0, len(batch), self.microbatch_size)]

    def gather_outputs(self, outputs: List[Any]) -> Any:
        # Concatenate or merge microbatch outputs
        return torch.cat(outputs, dim=0)
