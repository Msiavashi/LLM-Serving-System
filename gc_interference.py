import multiprocessing as mp
import time
import gc
import os
import signal
import sys

PHASE_INTERVAL = 30     # Seconds between phases
PHASE_DURATION = 3       # Duration of interference phase (seconds)
NUM_WORKERS = os.cpu_count()  # One per CPU core
ALLOC_SIZE = 1000000     # Number of 1KB objects per process

def worker(phase_event, shutdown_event):
    while not shutdown_event.is_set():
        if phase_event.is_set():
            # Interference phase: allocate & free lots of memory, trigger GC
            big_list = [bytearray(1024) for _ in range(ALLOC_SIZE // NUM_WORKERS)]
            del big_list
            gc.collect()
        else:
            # Idle phase: short sleep to reduce CPU usage
            time.sleep(0.01)

def start_workers(num_workers, phase_event, shutdown_event):
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(phase_event, shutdown_event))
        p.start()
        processes.append(p)
    return processes

def stop_workers(processes):
    print("Stopping workers...")
    for p in processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
    print("All workers stopped.")

def phased_interference():
    phase_event = mp.Event()
    shutdown_event = mp.Event()

    processes = start_workers(NUM_WORKERS, phase_event, shutdown_event)

    def handle_exit(signum, frame):
        print("\nExiting... (caught signal {})".format(signum))
        shutdown_event.set()
        phase_event.set()  # Wake workers if sleeping
        stop_workers(processes)
        sys.exit(0)

    # Ensure clean shutdown on SIGINT/SIGTERM
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print(f"Running phased GC interference using {NUM_WORKERS} worker processes.")
    try:
        while True:
            print(">>> Interference phase START")
            phase_event.set()
            time.sleep(PHASE_DURATION)
            print("<<< Interference phase END")
            phase_event.clear()
            time.sleep(PHASE_INTERVAL - PHASE_DURATION)
    except KeyboardInterrupt:
        handle_exit(signal.SIGINT, None)
    except Exception as e:
        print("Unexpected error:", e)
        handle_exit(signal.SIGTERM, None)

if __name__ == "__main__":
    phased_interference()
