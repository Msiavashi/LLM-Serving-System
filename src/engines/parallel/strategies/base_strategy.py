class BaseParallelismStrategy:
    """
    Base class for all parallelism strategies.
    """
    def __init__(self, dist_manager=None):
        self.dist_manager = dist_manager
        self.parallelize_plan = {}

    def set_parallelize_plan(self, plan):
        """Set the parallelization plan (function or dict)"""
        self.parallelize_plan = plan

    def parallelize_model(self, model):
        raise NotImplementedError("parallelize_model must be implemented in subclasses.")
