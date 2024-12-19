import random
import sys
import os
from typing import List
from mpi4py import MPI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from src.schedulers.round_robin_scheduler import RoundRobinScheduler, ModelInstance
from src.models.mixtral_queue_model import MyCustomMixtral


def initialize_model_and_tokenizer(rank: int):
    config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    if (tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Initialize single model on this rank's GPU
    device_map = {'': f'cuda:{rank}'}
    model = MyCustomMixtral.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        config=config,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    return [ModelInstance(model, f"cuda:{rank}")], tokenizer


def usage_example():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Initialize model on this rank
    models, tokenizer = initialize_model_and_tokenizer(rank)
    
    if rank == 0:
        # Create sample prompts only on rank 0
        prompts = [
            "Tell me about artificial intelligence.",
            "Explain how solar panels work.",
            "Describe the water cycle.",
            "What is DNA?",
            "How do electric cars function?",
            "Explain blockchain technology.",
            "What causes climate change?",
            "How does the human brain work?",
            "Describe quantum entanglement.",
            "What is machine learning?",
            "How do smartphones work?",
            "Explain evolution.",
            "What is dark matter?",
            "How do vaccines work?",
            "Describe plate tectonics.",
            "What is cryptocurrency?",
            "How does photosynthesis work?",
            "Explain nuclear fusion.",
            "What is artificial neural network?",
            "How do computers process data?",
            "Describe black holes.",
            "What is renewable energy?",
            "How does the internet work?",
            "Explain quantum computing.",
            "What is gene editing?",
            "How do airplanes fly?",
            "Describe cloud computing.",
            "What is virtual reality?",
            "How does 5G technology work?",
            "Explain artificial photosynthesis.",
            # Adding more diverse topics
            "What is nanotechnology?",
            "How do batteries store energy?",
            "Describe quantum tunneling.",
            "What is machine vision?",
            "How do robots navigate?",
            "Explain superconductivity.",
            "What is quantum cryptography?",
            "How does GPS work?",
            "Describe neural plasticity.",
            "What is fusion energy?",
            "How do touch screens work?",
            "Explain dark energy.",
            "What is quantum supremacy?",
            "How do self-driving cars work?",
            "Describe quantum teleportation.",
            "What is edge computing?",
            "How do fiber optics work?",
            "Explain CRISPR technology.",
            "What is quantum sensing?",
            "How do nuclear reactors work?",
            # Continue with more science and tech topics...
            "Describe quantum radar.",
            "What is biomimicry?",
            "How do smart grids work?",
            "Explain quantum mechanics.",
            "What is synthetic biology?",
            "How do rocket engines work?",
            "Describe quantum dots.",
            "What is deep learning?",
            "How do wind turbines work?",
            "Explain string theory.",
            "What is augmented reality?",
            "How do LED lights work?",
            "Describe quantum algorithms.",
            "What is biotechnology?",
            "How do MRI machines work?",
            "Explain quantum error correction.",
            "What is IoT technology?",
            "How do semiconductors work?",
            "Describe quantum metrology.",
            "What is cognitive computing?",
            "How do radio waves work?",
            "Explain quantum chemistry.",
            "What is neuromorphic computing?",
            "How do lasers work?",
            "Describe quantum biology.",
            "What is space propulsion?",
            "How do fuel cells work?",
            "Explain quantum gravity.",
            "What is brain-computer interface?",
            "How do quantum sensors work?",
            # Additional technical topics
            "What is quantum communication?",
            "How do microprocessors work?",
            "Describe quantum materials.",
            "What is swarm robotics?",
            "How do digital cameras work?",
            "Explain quantum thermodynamics.",
            "What is molecular computing?",
            "How do hydroelectric dams work?",
            "Describe quantum networks.",
            "What is computer vision?",
            "How do particle accelerators work?",
            "Explain quantum simulation.",
            "What is bioinformatics?",
            "How do memory chips work?",
            "Describe quantum logic gates.",
            "What is cognitive robotics?",
            "How do neural networks learn?",
            "Explain quantum entanglement swapping.",
            "What is quantum machine learning?",
            "How do optical computers work?",
            "Describe quantum error detection.",
            "What is bio-inspired computing?",
            "How do quantum memories work?",
            "Explain topological computing.",
            "What is quantum annealing?",
            "How do quantum repeaters work?",
            "Describe molecular machines.",
            "What is quantum tomography?",
            "How do quantum clocks work?",
            "Explain quantum interference.",
            # Final set to reach 128
            "What is quantum holography?",
            "How do quantum random numbers work?",
            "Describe quantum teleportation protocols.",
            "What is quantum key distribution?",
            "How do quantum computers debug?",
            "Explain quantum phase estimation.",
            "What is quantum error mitigation?",
            "How do quantum gates work?",
            "Describe quantum coherence.",
            "What is quantum internet?",
            "How do quantum measurements work?",
            "Explain quantum superposition.",
            "What is quantum state tomography?",
            "How do quantum algorithms scale?",
            "Describe quantum information theory.",
            "What is quantum Shannon theory?",
            "How do quantum channels work?",
            "Explain quantum complexity theory.",
        ]
        random.shuffle(prompts)
        
        # Distribute prompts across ranks
        chunks = [prompts[i::size] for i in range(size)]
    else:
        chunks = None
    
    # Scatter prompts to all ranks
    local_prompts = comm.scatter(chunks, root=0)
    
    # Create and use the scheduler with local model
    scheduler = RoundRobinScheduler(models, tokenizer, batch_size=32)
    
    for prompt in local_prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    local_results = scheduler.run_scheduler()
    
    # Gather results back to rank 0
    all_results = comm.gather(local_results, root=0)
    
    # Print results only on rank 0
    if rank == 0:
        for rank_results in all_results:
            for seq in rank_results:
                print(f"\nPrompt: {seq.prompt}")
                print(f"Generated on device: {seq.device}")
                print(f"Generated text: {seq.get_generated_text(tokenizer)}")

if __name__ == "__main__":
    usage_example()
