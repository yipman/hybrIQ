# HybrIQ 🧠⚛️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0%2B-orange)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.46.0%2B-blueviolet)](https://qiskit.org/)

## Introduction

HybrIQ implements a hybrid classical-quantum approach for training and inference of a large language model (LLM) like GPT-2. It combines PyTorch for classical computations with Qiskit for quantum computations, focusing on matrix operations that could potentially be accelerated on a quantum computer.

## Key Features

- **Hybrid Design**: Integrates classical GPU-based processing with quantum simulation for specific operations
- **Quantum Components**: Implements quantum algorithms for inner product estimation (attention mechanism) and matrix-vector multiplication (feed-forward layers)
- **Simulation-Based**: Runs on IBM's quantum simulator via Qiskit, so no real quantum hardware is needed
- **Comprehensive Benchmarking**: Built-in benchmarking suite to compare quantum vs. classical performance across various parameters
- **Performance Visualization**: Generate plots and reports to analyze how quantum acceleration scales with model size, layers, quantum noise, and shot counts

> This project is an educational tool and research prototype, showcasing how quantum computing might enhance machine learning workflows, particularly for transformer architectures.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- Qiskit 0.46.0 or higher
- NumPy

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yipman/hybrIQ.git
   cd hybrid-transformer
   ```

2. **Install dependencies**:
   ```bash
   pip install torch==2.0.0 qiskit==0.46.0 numpy matplotlib tqdm pandas
   ```

3. **(Optional) Use a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Quantum Simulator

The code uses Qiskit's Aer simulator to emulate quantum circuits. You don't need access to a physical quantum computer to run this project.

## Usage

### Running a Forward Pass

The project includes a simplified transformer model with quantum-accelerated operations. Here's how to run the forward pass with a dummy input:

1. **Import the modules**:
   ```python
   import torch
   from model import Model, forward_pass
   ```

2. **Initialize the model**:
   ```python
   model = Model(vocab_size=100, d_model=4, n_layers=2, d_ff=16)
   ```

3. **Create a dummy input**:
   ```python
   batch = torch.zeros(2, 3, dtype=torch.long)  # Batch size 2, sequence length 3
   ```

4. **Run the forward pass**:
   ```python
   logits = forward_pass(model, batch)
   print(f"Logits shape: {logits.shape}")
   ```

### Quantum Functions

- **`quantum_inner_product(u, v)`**: Estimates the inner product <u|v> using the Hadamard test. This is used in the attention mechanism.
- **`quantum_matrix_vector_mult(W, x)`**: Performs matrix-vector multiplication W @ x with a quantum circuit. This is used in feed-forward layers.

These functions run on Qiskit's qasm_simulator and are designed for small-scale demonstration.

## Benchmarking

HybrIQ includes a comprehensive benchmarking suite to evaluate quantum vs. classical performance. The benchmarks measure:

- **Size Scaling**: How performance scales with model dimension
- **Layer Scaling**: How performance scales with transformer layers
- **Noise Sensitivity**: How quantum noise affects results
- **Shot Count Effects**: Trade-offs between accuracy and performance with different quantum shot counts

### Running Benchmarks

You can run benchmarks using the command-line interface:

```bash
# Quick benchmarks for testing
python benchmark.py

# Full comprehensive benchmark suite
python benchmark.py --full

# Specific benchmark types
python benchmark.py --size    # Only run size scaling benchmark
python benchmark.py --layers  # Only run layer scaling benchmark
python benchmark.py --shots   # Only run shot count benchmark
python benchmark.py --noise   # Only run noise sensitivity benchmark
```

### Benchmark Results

The benchmarking suite automatically generates:
- JSON files with raw performance data
- Visualizations showing performance comparisons
- CSV reports summarizing all benchmark results

Results are saved to `/d:/HybrIQ/benchmark_results/` by default.

## Project Structure

```
HybrIQ/
│
├── model.py             # Core hybrid model and quantum functions
├── benchmark.py         # Benchmarking suite for performance evaluation
├── quantum_utils.py     # Utilities for quantum circuit manipulation
├── requirements.txt     # Dependency list
└── README.md            # This documentation
```

- **model.py**: Contains the hybrid transformer implementation, including quantum and classical components.
- **benchmark.py**: Comprehensive benchmarking suite with visualization capabilities.
- **quantum_utils.py**: Helper functions for quantum circuit analysis and visualization.
- **requirements.txt**: Lists all required packages for easy installation.

## Contributing

I'd love for others to help improve this project! To contribute:

1. Fork the repository
2. Create a branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m 'Add new feature'`)
4. Push your branch (`git push origin feature-branch`)
5. Submit a pull request with a clear description of your changes

Please ensure your code is well-documented and aligns with the project's style.

## Reporting Issues

Found a bug or have an idea? Please report it via the GitHub issue tracker. Include:

- A detailed description of the issue
- Steps to reproduce it
- Any error messages or logs

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

### Additional Notes

**Current Limitations**: The quantum components use small dimensions (e.g., 4D vectors with 2 qubits) for simulation purposes. Scaling to real-world LLMs (e.g., 768 dimensions) would require advanced quantum hardware.

**Purpose**: This is an educational and experimental project, not a production-ready solution. Quantum acceleration of LLMs is still an emerging field!
