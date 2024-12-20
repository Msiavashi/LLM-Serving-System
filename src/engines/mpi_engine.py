from mpi4py import MPI
from typing import List, Any, Tuple, Callable
from datetime import datetime
import time

class MPIEngine:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def distribute_data(self, data: List[Any]) -> List[Any]:
        """Distribute data across ranks"""
        if self.rank == 0:
            chunks = [data[i::self.size] for i in range(self.size)]
        else:
            chunks = None
        return self.comm.scatter(chunks, root=0)

    def gather_results(self, local_results: Any) -> List[Any]:
        """Gather results from all ranks"""
        return self.comm.gather(local_results, root=0)

    def synchronize(self):
        """Synchronize all processes"""
        self.comm.Barrier()

    def run_with_timing(self, process_fn: Callable, models: List[Any], tokenizer: Any) -> Tuple[Any, float]:
        """Run a process with timing"""
        self.synchronize()
        total_start_time = time.time()
        start_timestamp = datetime.now()
        
        if self.rank == 0:
            print(f"\nTotal execution starting at: {start_timestamp}")
        
        # Per-rank timing
        rank_start_time = time.time()
        print(f"Rank {self.rank} starting at: {datetime.now()}")
        
        # Run the actual process
        result = process_fn(models, tokenizer)
        
        # Record completion time
        rank_end_time = time.time()
        rank_duration = rank_end_time - rank_start_time
        print(f"Rank {self.rank} finished at: {datetime.now()}")
        print(f"Rank {self.rank} total execution time: {rank_duration:.2f} seconds")
        
        # Gather results and synchronize
        all_results = self.gather_results(result)
        self.synchronize()
        
        total_duration = time.time() - total_start_time
        
        if self.rank == 0:
            print(f"\nTotal execution finished at: {datetime.now()}")
            print(f"Total execution time: {total_duration:.2f} seconds")
        
        return all_results, total_duration
