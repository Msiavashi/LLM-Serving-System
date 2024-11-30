# LLM Serving System

The LLM Serving System is designed to facilitate the deployment and inference of large language models (LLMs) optimized for Mixture of Experts (MoE) models. The framework emphasizes modularity, efficiency, and performance. Its modular design allows researchers to quickly develop and test their ideas with minimal effort.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. Install the required dependencies using:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Running Examples

Navigate to the `examples` directory and run any of the example scripts. For instance, to run the `offline_inference_mixtral.py` example:

```bash
python examples/offline_inference_mixtral.py
```

### Directory Structure

- `examples/`: Example scripts for different MoE models.
- `src/`: Core source code for sequence handling, performance metrics, and schedulers.
- `tests/`: Unit and integration tests for the LLM Serving System.
- `requirements.txt`: Dependency list.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License.