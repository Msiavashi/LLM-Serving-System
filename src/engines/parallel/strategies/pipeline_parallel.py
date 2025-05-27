# class PipelineParallelStrategy:
#     def __init__(self, device_type='cuda', chunks=1):
#         self.device_type = device_type
#         self.chunks = chunks

#     def setup(self, rank, world_size):
#         import torch.distributed as dist
#         self.rank = rank
#         self.world_size = world_size
#         self.device = torch.device(f'{self.device_type}:{self.rank}')
#         torch.cuda.set_device(self.device)
#         dist.init_process_group("nccl", rank=rank, world_size=world_size)

#     def init_model_shard(self, model_partitions, parallelize_plan=None):
#         # model_partitions should be a list of nn.Sequential, one per stage (rank)
#         assert len(model_partitions) == self.world_size, "Model must be split for each stage"
#         my_stage = model_partitions[self.rank].to(self.device)
#         self.model = Pipe(my_stage, chunks=self.chunks, checkpoint="never", device=self.device)

#     def forward(self, batch):
#         batch = batch.to(self.device)
#         # Pipe expects input shape: [microbatches, ...] if chunks > 1
#         return self.model(batch)
