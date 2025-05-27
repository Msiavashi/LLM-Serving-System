from abc import ABC, abstractmethod

class ParallelStrategy(ABC):
    def __init__(self, device_type='cuda'):
        self.device_type = device_type
        self.rank = None
        self.world_size = None
        self.device = None
        self.model = None

    @abstractmethod
    def setup(self, rank, world_size):
        """
        Sets up distributed process group, device mesh, etc.
        """
        pass

    @abstractmethod
    def init_model_shard(self, model, parallelize_plan=None):
        """
        Initializes the model for this worker/rank (shard/replica/stage).
        For TensorParallel: uses parallelize_plan to shard model.
        For DataParallel: simple replica.
        For PipelineParallel: partitioned model.
        """
        pass

    @abstractmethod
    def forward(self, batch):
        """
        Runs a forward pass on the model according to the parallelism plan.
        """
        pass
