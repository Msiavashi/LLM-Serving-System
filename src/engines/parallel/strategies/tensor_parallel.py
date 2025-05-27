import torch
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import init_device_mesh

from src.models.model_factory import ModelFactory

class TensorParallelStrategy:
    def __init__(self, device_type='cuda', parallelize_plan=None):
        self.device_type = device_type
        self.parallelize_plan = parallelize_plan

    def setup(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'{self.device_type}:{self.rank}')
        torch.cuda.set_device(self.device)
        self.mesh = init_device_mesh(self.device_type, [self.world_size])

    def init_model_shard(self, model_checkpoint: str, parallelize_plan):
        model = ModelFactory.create_model(model_checkpoint, device=self.device).model
        self.model = parallelize_module(model, self.mesh, parallelize_plan())
        self.model.to(self.device) # TODO: I guess the model is already allocated on the device.

    def forward(self, batch):
        # Each rank works on its partitioned model
        batch = batch.to(self.device)
        return self.model(batch)
