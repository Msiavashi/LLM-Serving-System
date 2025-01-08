import os
import mmap
from mpi4py import MPI
from src.queues.fcfs_queue import FCFSQueue

class MPISharedQueueManager:
    queues = {}
    shared_memory_regions = {}
    comm = MPI.COMM_WORLD  # Add MPI communicator

    @staticmethod
    def add_queues(rank, queue_num, layer):
        if layer not in MPISharedQueueManager.queues:
            MPISharedQueueManager.queues[layer] = {}
            MPISharedQueueManager.shared_memory_regions[layer] = {}
        
        MPISharedQueueManager.queues[layer][rank] = []
        MPISharedQueueManager.shared_memory_regions[layer][rank] = []
        
        for _ in range(queue_num):
            # Create shared memory region
            shm_size = 1024 * 1024  # 1MB per queue, adjust as needed
            fd = os.open(f'/dev/shm/mpi_queue_{layer}_{rank}_{_}', os.O_CREAT | os.O_RDWR)
            os.ftruncate(fd, shm_size)
            
            # Memory map the shared region
            shm = mmap.mmap(fd, shm_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)
            
            queue = FCFSQueue()
            queue.shared_memory = shm
            
            MPISharedQueueManager.queues[layer][rank].append(queue)
            MPISharedQueueManager.shared_memory_regions[layer][rank].append(shm)

        # Add barrier to ensure all processes have created their queues
        MPISharedQueueManager.comm.Barrier()

    @staticmethod
    def get(rank, layer, queue_idx):
        # Add barrier to ensure queues are ready before access
        MPISharedQueueManager.comm.Barrier()
        return MPISharedQueueManager.queues[layer][rank][queue_idx]

    @staticmethod
    def get_queues_at_layer(rank, layer):
        # Add barrier to ensure queues are ready before access
        MPISharedQueueManager.comm.Barrier()
        return MPISharedQueueManager.queues[layer][rank]

    @staticmethod
    def cleanup():
        # Add barrier to ensure all processes are done before cleanup
        MPISharedQueueManager.comm.Barrier()
        for layer in MPISharedQueueManager.shared_memory_regions:
            for rank in MPISharedQueueManager.shared_memory_regions[layer]:
                for shm in MPISharedQueueManager.shared_memory_regions[layer][rank]:
                    shm.close()
                    try:
                        os.unlink(f'/dev/shm/mpi_queue_{layer}_{rank}')
                    except:
                        pass