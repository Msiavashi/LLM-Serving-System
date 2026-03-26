# QLLM: Priority-Aware Preemptive Scheduling for MoE Inference

QLLM is an inference system for Mixture of Experts (MoE) large language models that enables **fine-grained, expert-level preemption** and **priority-aware scheduling**. It reduces latency-sensitive (LS) time-to-first-token by up to 101.6x while maintaining throughput, by breaking the rigid iteration-level scheduling used by existing systems.

> **Paper**: [Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference](https://doi.org/10.1145/3721146.3721956)
> Published at **EuroMLSys '25** (5th Workshop on Machine Learning and Systems), March 2025.
> Authors: Mohammad Siavashi, Faezeh Keshmiri Dindarloo, Dejan Kostic, Marco Chiesa — KTH Royal Institute of Technology

## Key Features

- **Expert-Level Preemption**: Per-expert FIFO queues enable preemption within MoE layers, not just between iterations
- **Priority-Aware Scheduling**: 4-queue system (LS/BE x prefill/decode) with strict priority ordering
- **Unified Dynamic Cache**: Decoupled sequence-level KV cache management with optional int8 compression
- **Modular Architecture**: Sequence/Batch abstractions integrate with HuggingFace models via minimal modifications
- **Multiple Schedulers**: FCFS, Priority (with preemption), Round-Robin (multi-GPU)
- **Online Serving**: FastAPI server with Redis-backed queues and OpenAI-compatible API

## Supported Models

| Model | Standard | Queue (per-expert) |
|---|---|---|
| Mixtral 8x7B | `mixtral` | `mixtral_queue` |
| Phi-3.5-MoE | `phimoe` | `phimoe_queue` |
| Llama 3 8B | `llama3_8b` | — |
| Switch Transformers | via examples | — |

All models are loaded with 4-bit quantization (BitsAndBytes) by default.

## Requirements

- Python >= 3.10
- CUDA >= 12.0 with a compatible GPU (tested on A100 80GB, H100, L40)
- ~23 GB GPU memory for Mixtral 8x7B (4-bit quantized)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: `flash-attn` requires a CUDA-capable GPU and may take several minutes to build from source. If you encounter issues, install it separately:

```bash
pip install flash-attn --no-build-isolation
```

Optional dependencies for distributed inference:

```bash
pip install mpi4py  # requires MPI (e.g., OpenMPI) installed on the system
```

## Quick Start

### Offline Inference (Llama 3 8B)

```bash
python examples/offline_llama_8b.py
```

### Offline Inference with Priority Scheduling (Mixtral)

```bash
python examples/offline_inference_mixtral.py
```

This example uses the priority scheduler with per-expert queuing — the core QLLM contribution. 20% of requests are marked as latency-sensitive (LS) and receive priority scheduling.

### Offline Inference with Dataset

```bash
python examples/offline_inference_with_dataset.py
```

Uses prompts from the ShareGPT dataset with the Mixtral queue model.

### Online Inference (Experimental)

Start the server stack:

```bash
# 1. Start Redis
redis-server

# 2. Start the scheduler service
python -m src.services.runner

# 3. Start the API server (standard or priority-aware)
python -m src.server.api_server              # FCFS mode
python -m src.server.api_server_with_priority # Priority mode

# 4. Send requests
python examples/async_inference_example.py
```

The API server exposes an OpenAI-compatible `/v1/chat/completions` endpoint with an additional `priority` field (0 = best-effort, 1 = latency-sensitive).

### Multi-GPU Inference

```bash
mpirun -n <num_gpus> python examples/multi_gpu_example.py
```

## Architecture (v2)

QLLM v2 uses **composition over inheritance** — it loads any HuggingFace model unmodified and injects per-expert queue scheduling at runtime:

```
Scheduler (FCFS / Priority)
    ↓
QllmEngine
    ├── ModelAdapter (loads any HF model via AutoModelForCausalLM)
    │   ├── Auto-detects MoE layers
    │   └── Injects QueueAwareMoEWrapper at runtime (no subclassing)
    ├── SequenceCacheManager (HF-native DynamicCache per sequence)
    └── SamplingProcessor (vectorized, configurable temperature/top-k/top-p/EOS)
```

**Key design**: MoE blocks are wrapped via `setattr()` module replacement — the original expert networks and gate are reused directly, preserving all HF optimizations (SDPA, quantization, etc.).

### New API (v2)

```python
from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.samplers.sampling_params import SamplingParams

adapter = ModelAdapter.from_name("mixtral", rank=0)  # Any HF model
adapter.inject_queues()  # Auto-detect MoE, inject queue wrappers

engine = QllmEngine(
    model_adapter=adapter,
    sampling_params=SamplingParams(temperature=0.7, eos_token_id=adapter.tokenizer.eos_token_id),
)
```

### Directory Structure

```
src/
├── batching/          # Batch abstraction and policies
├── cache/             # SequenceCacheManager (HF-native), compression, dynamic cache
├── config/            # YAML configuration management
├── engines/           # QllmEngine (v2), ModelEngine (legacy), MPI engine
├── mixins/            # QueueAwareMoEWrapper (v2), legacy model I/O mixins
├── models/            # ModelAdapter (v2), legacy model classes, factory
├── monitoring/        # Performance metrics (TTFT, TPOT, throughput)
├── queues/            # FCFS queue with memory/Redis storage backends
├── samplers/          # SamplingParams, SamplingProcessor, SamplingMetadata
├── schedulers/        # FCFS, Priority, Round-Robin schedulers
├── sequence/          # Sequence abstraction (state, KV cache, timing)
├── server/            # FastAPI servers (standard and priority-aware)
└── services/          # Scheduler service and factory
examples/              # Runnable examples (v1 legacy + v2 new architecture)
tests/                 # Unit and integration tests
benchmarks/            # Reproducible benchmark harness
```

## Results

### Paper results (Mixtral 8x7B, A100 80GB, ShareGPT):

| Metric | Result |
|---|---|
| LS TTFT reduction | up to 101.6x (avg 65.2x) |
| SLO compliance (3s) | up to 7 req/s (baseline fails) |
| LS turnaround time | up to 12.8x reduction |
| Throughput | comparable or better than baseline |

### v2 architecture improvements (vs v1 baseline):

| Metric | v1 (baseline) | v2 (rearchitected) | Change |
|---|---|---|---|
| Mixtral decode TPOT | 1.25s | 0.325s | **3.8x faster** |
| Mixtral throughput | 50 tok/s | 91 tok/s | **1.8x higher** |
| Llama 8B decode TPOT | 0.068s | 0.057s | **16% faster** |
| Tokens per sequence | 10 (bug) | 20 (fixed) | Correct |
| New model effort | Subclass per model | Zero (auto-detect) | Eliminated |

## Citation

```bibtex
@inproceedings{siavashi2025qllm,
  title={Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference},
  author={Siavashi, Mohammad and Dindarloo, Faezeh Keshmiri and Kostic, Dejan and Chiesa, Marco},
  booktitle={Proceedings of the 5th Workshop on Machine Learning and Systems (EuroMLSys '25)},
  pages={132--138},
  year={2025},
  publisher={ACM},
  doi={10.1145/3721146.3721956}
}
```

## License

This project is licensed under the MIT License.
