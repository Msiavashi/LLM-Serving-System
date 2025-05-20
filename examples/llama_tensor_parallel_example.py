import sys
import os
from time import sleep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.parallel.model_parallelism import ModelParallelismManager
from src.engines.parallel.distributed_manager import DistributedManager
from src.models.model_factory import ModelFactory
from src.engines.distributed_engine import DistributedEngine
from src.engines.parallel.parallelization_plans.llama8b_tensor_parallel import llama8b_tensor_parallel_plan

# Create a distributed manager with a Tensor Parallel strategy
dist_manager = DistributedManager()
parallelism_manager = ModelParallelismManager(dist_manager=dist_manager)
strategy = parallelism_manager.create_tensor_parallel_strategy(name="tensor_parallelism")
strategy.set_parallelize_plan(llama8b_tensor_parallel_plan)

model_instance, tokenizer = ModelFactory.create_model(
    model_type="llama3_8b",
    dist_manager=dist_manager,
    strategy=strategy
)

engine = DistributedEngine(
    model_stages=[model_instance.model],
    strategy=strategy,
    dist_manager=dist_manager
)