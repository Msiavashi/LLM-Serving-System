from src.monitoring.performance_monitor import PerformanceMonitor
from src.queues import FCFSQueue as SequenceQueue
class ModelInstance:
    def __init__(self, model, device):
        self.model = model  # Don't move the model
        self.device = device  # Just store target device for sequences
        self.prefill_queue = SequenceQueue()
        self.decode_queue = SequenceQueue()
        self.prefill_stats = {"tokens": 0, "time": 0}
        self.decode_stats = {"tokens": 0, "time": 0}
        self.finished_sequences = []
        self.monitor = PerformanceMonitor()
