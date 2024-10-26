from abc import ABC, abstractmethod

class BaseBatchPolicy(ABC):
    
    @abstractmethod
    def add_to_batch(self, sequence):
        pass
    
    @abstractmethod
    def get_next_batch(self):
        pass
