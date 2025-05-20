import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.model_factory import ModelFactory
from src.engines.distributed_engine import DistributedEngine
from src.engines.parallel.strategies. import PipelineParallelStrategy
from src.engines.parallel.distributed_manager import DistributedManager

def main():
    # Distributed setup
    dist_manager = DistributedManager()
    rank = dist_manager.rank
    world_size = dist_manager.world_size

    # Model and strategy setup
    model_type = "llama3_8b"
    num_stages = world_size
    microbatch_size = 2
    tensor_shape = (microbatch_size, 128)  # Example shape

    # Load model and split into stages (user must implement split_model)
    _, tokenizer = ModelFactory.create_model(model_type, rank=rank)
    model = ... # Load the full model here
    strategy = PipelineParallelStrategy(num_stages=num_stages, microbatch_size=microbatch_size, tensor_shape=tensor_shape)
    model_stages = strategy.split_model(model)  # User must implement this

    # Engine
    engine = DistributedEngine(model_stages, strategy, dist_manager)

    # Dummy batch
    batch = torch.randn((microbatch_size * 4, 128), device=dist_manager.device)

    # Run
    output = engine.run_batch(batch)
    if rank == 0:
        print("Pipeline output:", output)

    dist_manager.finalize()

if __name__ == "__main__":
    main()
