import torch

class DataParallelStrategy:
    def __init__(self, device_type='cuda'):
        self.device_type = device_type

    def setup(self, rank, world_size):
        import torch.distributed as dist
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'{self.device_type}:{self.rank}')
        torch.cuda.set_device(self.device)
        # Each worker initializes a default process group for communication
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def init_model_shard(self, model, parallelize_plan=None):
        # Each worker gets a full replica of the model (no tensor sharding)
        self.model = model.to(self.device)

    def forward(self, batch):
        # Each worker works on its slice of data (make sure batch is partitioned upstream)
        batch = batch.to(self.device)
        return self.model(batch)
