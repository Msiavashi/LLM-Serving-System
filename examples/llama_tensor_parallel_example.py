import sys
import os
import signal
import time

# Add proper signal handling
signal.signal(signal.SIGINT, signal.default_int_handler)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.parallel.distributed_manager import DistributedManager
from src.models.model_factory import ModelFactory
from src.engines.parallel.strategies.tensor_parallel import TensorParallelStrategy
from src.engines.parallel.parallelization_plans.llama8b_tensor_parallel import llama8b_tensor_parallel_plan

def main():
    try:
        print("Initializing distributed manager...")
        dist_manager = DistributedManager()

        print("Loading tokenizer...")
        tokenizer = ModelFactory.get_tokenizer_for(model_name="llama3_8b")

        # Just pass the model checkpoint name, don't create actual model object
        model_checkpoint = "llama3_8b"

        print("Creating tensor parallel strategy...")
        strategy = TensorParallelStrategy(
            device_type='cuda',
            parallelize_plan=llama8b_tensor_parallel_plan
        )

        print("Creating and initializing distributed workers...")
        dist_manager.create_workers(model_checkpoint, strategy)
        print("Distributed workers created successfully.")
        
        print("Workers are running. Press Ctrl+C to exit...")
        while True:
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down gracefully...")
        if 'dist_manager' in locals():
            dist_manager.shutdown()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

