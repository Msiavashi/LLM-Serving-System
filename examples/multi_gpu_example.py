import random
import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.mpi_engine import MPIEngine
from src.models.model_factory import ModelFactory
from src.schedulers.round_robin_scheduler import RoundRobinScheduler


def get_example_prompts() -> List[str]:
    """Returns a list of sample prompts"""
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
    return prompts

def process_prompts(models, tokenizer, prompts, rank):
    """Process prompts using the scheduler"""
    scheduler = RoundRobinScheduler(models, tokenizer, batch_size=32, rank=rank)
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    return scheduler.run_scheduler()

def usage_example():
    # Initialize the MPI engine
    engine = MPIEngine()
    
    # Initialize model and tokenizer for this rank
    models, tokenizer = ModelFactory.create_mixtral_model(engine.rank)
    
    # Prepare prompts on rank 0
    if engine.rank == 0:
        prompts = get_example_prompts()
    else:
        prompts = None
    
    # Distribute prompts across ranks
    local_prompts = engine.distribute_data(prompts)
    
    # Run the processing with timing
    all_results, _ = engine.run_with_timing(
        lambda m, t: process_prompts(m, t, local_prompts, engine.rank),
        models,
        tokenizer
    )
    
    # Print results on rank 0
    if engine.rank == 0:
        for rank_results in all_results:
            for seq in rank_results:
                print(f"\nPrompt: {seq.prompt}")
                print(f"Generated on device: {seq.device}")
                print(f"Generated text: {seq.get_generated_text(tokenizer)}")

if __name__ == "__main__":
    usage_example()
