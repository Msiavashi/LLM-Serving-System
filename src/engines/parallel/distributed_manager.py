import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional
import logging
import signal
import sys


class DistributedManager:
    def __init__(
        self,
        backend: str = "nccl",
        world_size: Optional[int] = None,
        init_method: str = "env://",
    ):
        self.backend = backend
        self.world_size = world_size if world_size is not None else torch.cuda.device_count()
        self.init_method = init_method
        self.rank = 0  # Set in init_process_group

        # Set default environment variables for process group init
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        # Track initialized status
        self._initialized = False
        self.workers = []
        self.processes = []

    def _worker_entry(self, rank, world_size, model_name, backend, init_method):
        """Worker process entry point - receives model name instead of model object to avoid pickling issues"""
        import signal
        import sys
        import torch
        import torch.distributed as dist

        # Set up signal handler for clean shutdown
        def sigint_handler(signum, frame):
            print(f"[Rank {rank}] Received SIGINT, exiting.")
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        try:
            print(f"[Rank {rank}] Setting device and initializing process group...")
            torch.cuda.set_device(rank)
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
            print(f"[Rank {rank}] Process group initialized.")

            # Import here to avoid pickle issues
            from src.engines.workers.gpu_worker import GPUWorker
            from src.models.model_factory import ModelFactory
            from src.engines.parallel.strategies.tensor_parallel import TensorParallelStrategy
            from src.engines.parallel.parallelization_plans.llama8b_tensor_parallel import llama8b_tensor_parallel_plan

            # Create strategy and model inside the worker process
            strategy = TensorParallelStrategy(
                device_type='cuda',
                parallelize_plan=llama8b_tensor_parallel_plan
            )

            # Setup strategy
            strategy.setup(rank, world_size)

            # Initialize model in worker process
            strategy.init_model_shard(model_name, llama8b_tensor_parallel_plan)

            # Create worker with initialized model
            worker = GPUWorker(rank, world_size, strategy.model, strategy)
            print(f"[Rank {rank}] Worker created successfully")

            # Keep process alive
            import time
            while True:
                time.sleep(10)

        except KeyboardInterrupt:
            print(f"[Rank {rank}] KeyboardInterrupt received, shutting down.")
            sys.exit(0)
        except Exception as e:
            print(f"[Rank {rank}] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def create_workers(self, model_name, strategy):
        """Create worker processes - passing model_name instead of model object"""
        world_size = self.world_size
        backend = self.backend
        init_method = self.init_method

        # Create context for spawning processes
        ctx = mp.get_context("spawn")
        self.processes = []
        self.workers = []

        print(f"Creating {world_size-1} worker processes...")

        # Spawn worker processes for ranks 1 to world_size-1
        for rank in range(1, world_size):
            p = ctx.Process(
                target=self._worker_entry,
                args=(rank, world_size, model_name, backend, init_method),
            )
            p.daemon = True  
            p.start()
            self.processes.append(p)
            print(f"Started worker process for rank {rank}")

        import torch.distributed as dist

        def sigint_handler(signum, frame):
            print("[Rank 0] Received SIGINT, terminating all processes.")
            for p in self.processes:
                p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        print("[Rank 0] Setting device and initializing process group...")
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=0,
        )
        print("[Rank 0] Process group initialized.")

        # Initialize model for rank 0
        strategy.setup(0, world_size)
        strategy.init_model_shard(model_name, llama8b_tensor_parallel_plan)

        # Create GPU worker for rank 0
        from src.engines.workers.gpu_worker import GPUWorker
        worker = GPUWorker(0, world_size, strategy.model, strategy)
        print("[Rank 0] Worker created successfully")
        self.workers.append(worker)

    def barrier(self):
        """Synchronize all processes"""
        if dist.is_initialized():
            dist.barrier()

    def shutdown(self):
        """Clean shutdown of all processes"""
        print("Shutting down distributed manager...")
        for p in self.processes:
            if p.is_alive():
                p.terminate()
        print("All worker processes terminated.")