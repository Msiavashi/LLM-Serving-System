from typing import Dict, Any, Optional
from src.engines.parallel.distributed_manager import DistributedManager
from src.engines.parallel.strategies.tensor_parallel import TensorParallelismStrategy

class ModelParallelismManager:
    """
    This class provides a centralized way to create, manage, and switch between different
    parallelism strategies for distributed model execution. It acts as an interface between
    model factory and various distributed execution strategies.
    Attributes:
        dist_manager (DistributedManager): The distributed manager handling process coordination.
        strategies (Dict[str, Any]): Dictionary storing all registered parallelism strategies.
        current_strategy (Any): Currently active parallelism strategy.
        _scaling_factor (int): Scaling factor used to determine degree of parallelism.
    Methods:
        create_tensor_parallel_strategy: Creates and registers a tensor parallelism strategy.
        set_active_strategy: Sets a specific registered strategy as active.
        get_parallel_config_for_model: Generates configuration dictionary for model initialization.
        set_scaling_factor: Sets the parallelism scaling factor.
        get_strategy: Retrieves a specific strategy or the current one.
        current_parallel_config: Property to get the current parallel configuration.
    """
    
    def __init__(self, dist_manager: Optional[DistributedManager] = None):
        """Initialize the model parallelism manager"""
        if dist_manager is None:
            self.dist_manager = DistributedManager()
        else:
            self.dist_manager = dist_manager
        
        self.strategies = {}
        self.current_strategy = None
        self._scaling_factor = 1
        
    def create_tensor_parallel_strategy(self, name: str = "default") -> TensorParallelismStrategy:
        """Create a tensor parallelism strategy"""
        strategy = TensorParallelismStrategy(dist_manager=self.dist_manager)
        self.strategies[name] = strategy
        
        if self.current_strategy is None:
            self.current_strategy = strategy
            
        return strategy
    
    def set_active_strategy(self, name: str) -> None:
        """Set the active parallelism strategy"""
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found")
        self.current_strategy = self.strategies[name]
    
    def get_parallel_config_for_model(self) -> Dict[str, Any]:
        """Generate parallel configuration dictionary for model initialization"""
        # This would include configuration settings for various model parallel techniques
        # such as tensor, pipeline, or expert parallelism
        config = {}
        
        # For tensor parallelism, we might set:
        if self._scaling_factor > 1:
            config["tensor_parallel_size"] = min(self.dist_manager.world_size, self._scaling_factor)
            
        return config
    
    def set_scaling_factor(self, factor: int) -> None:
        """Set the scaling factor for parallelism"""
        self._scaling_factor = max(1, factor)
        
        # Update all managed strategies
        for strategy in self.strategies.values():
            if hasattr(strategy, 'set_scaling_factor'):
                strategy.set_scaling_factor(factor)
    
    def get_strategy(self, name: str = None) -> Any:
        """Get a specific strategy or the current one"""
        if name is None:
            return self.current_strategy
        
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found")
        
        return self.strategies[name]
    
    @property
    def current_parallel_config(self) -> Dict[str, Any]:
        """Get the current parallel configuration"""
        return self.get_parallel_config_for_model()
