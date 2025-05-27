import torch

class GPUWorker:
    def __init__(self, rank, world_size, model_checkpoint, strategy):
        self.rank = rank
        self.world_size = world_size
        self.strategy = strategy
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)
        # Setup parallel environment
        self.strategy.setup(rank, world_size)
        # Parallelize and move model to device
        self.strategy.init_model_shard(model_checkpoint, self.strategy.parallelize_plan)

    def run(self, batch):
        return self.strategy.forward(batch)
