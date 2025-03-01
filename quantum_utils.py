"""
Quantum utilities for HybrIQ.
Provides helper functions for quantum execution and analysis.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendSampler, Sampler, BackendEstimator
import time
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger('HybrIQ.quantum_utils')

def execute_circuits(circuits, backend, shots=1024, optimization_level=3):
    """
    Execute quantum circuits using Qiskit 1.0 compatible execution pattern with primitives.
    
    Args:
        circuits: Single circuit or list of circuits to execute
        backend: Quantum backend to run on
        shots: Number of shots per circuit
        optimization_level: Transpiler optimization level
        
    Returns:
        Result object containing measurement counts
    """
    start_time = time.time()
    
    # Handle case of a single circuit
    if not isinstance(circuits, list):
        circuits = [circuits]
    
    # Create a sampler from the backend
    # Fix: Update to use the correct BackendSampler API
    sampler = BackendSampler(
        backend=backend,
        options={
            "optimization_level": optimization_level,
            "shots": shots
        }
    )
    
    # Run using the sampler primitive
    job = sampler.run(circuits)
    result = job.result()
    
    exec_time = time.time() - start_time
    logger.debug(f"Executed {len(circuits)} circuits in {exec_time:.4f}s")
    
    return result

def execute_with_sampler(circuits, backend, shots=1024):
    """
    Execute quantum circuits using the BackendSampler primitive.
    
    Args:
        circuits: Single circuit or list of circuits to execute
        backend: Quantum backend to run on
        shots: Number of shots per circuit
        
    Returns:
        Sampler job result
    """
    # Create a BackendSampler from the backend
    sampler = BackendSampler(backend=backend)
    
    # Run with the sampler
    job = sampler.run(circuits, shots=shots)
    return job.result()

def execute_with_estimator(circuits, observable, backend, shots=1024):
    """
    Execute quantum circuits and measure expectation values using the Estimator primitive.
    
    Args:
        circuits: Quantum circuit or list of circuits
        observable: Observable or list of observables to measure
        backend: Quantum backend to run on
        shots: Number of shots per circuit
        
    Returns:
        Estimator job result with expectation values
    """
    # Create a BackendEstimator from the backend
    estimator = BackendEstimator(
        backend=backend,
        run_options={"shots": shots}
    )
    
    # Run with the estimator
    job = estimator.run(circuits, observable)
    return job.result()

def estimate_quantum_resources(circuit):
    """
    Estimate quantum resources required by a circuit.
    
    Args:
        circuit: QuantumCircuit to analyze
        
    Returns:
        Dictionary with resource estimates
    """
    depth = circuit.depth()
    width = circuit.num_qubits
    gate_counts = circuit.count_ops()
    
    # Count specific gate types
    cx_count = gate_counts.get('cx', 0)
    measure_count = gate_counts.get('measure', 0)
    
    return {
        "depth": depth,
        "width": width,
        "gate_counts": gate_counts,
        "cx_count": cx_count,
        "measure_count": measure_count
    }

def quantum_resource_estimator(model, batch_size=2, seq_len=3, full_details=False):
    """
    Estimate quantum resources needed by the HybrIQ model.
    
    Args:
        model: HybrIQ Model instance
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        full_details: Whether to return full details
        
    Returns:
        Dictionary with resource estimates
    """
    # Import here to avoid circular imports
    from model import create_inner_product_circuit, create_matrix_vector_circuit
    
    # Get example circuits
    ip_circuit = create_inner_product_circuit(model.d_model)
    mv_circuit = create_matrix_vector_circuit(min(int(np.log2(model.d_model)), 5))
    
    ip_resources = estimate_quantum_resources(ip_circuit)
    mv_resources = estimate_quantum_resources(mv_circuit)
    
    # Estimate execution statistics
    n_attention_ops = batch_size * seq_len * seq_len * len(model.layers)  # Q*K operations in attention
    n_ff_ops = batch_size * seq_len * len(model.layers)  # Feed-forward operations
    
    # Estimate shots
    from model import config
    total_shots = (n_attention_ops + n_ff_ops) * config.shots
    
    result = {
        "model_dims": {
            "d_model": model.d_model,
            "n_layers": len(model.layers)
        },
        "circuit_resources": {
            "inner_product": ip_resources,
            "matrix_vector": mv_resources
        },
        "execution_stats": {
            "attention_operations": n_attention_ops,
            "feedforward_operations": n_ff_ops,
            "estimated_total_shots": total_shots
        }
    }
    
    if not full_details:
        # Simplified output for quick overview
        return {
            "d_model": model.d_model,
            "n_layers": len(model.layers),
            "inner_product_depth": ip_resources["depth"],
            "matrix_vector_depth": mv_resources["depth"],
            "total_operations": n_attention_ops + n_ff_ops,
            "total_shots": total_shots
        }
    
    return result

def compare_classical_quantum(model, batch, use_quantum=True):
    """
    Compare a single forward pass with both quantum and classical execution.
    
    Args:
        model: HybrIQ Model instance
        batch: Input batch
        use_quantum: Whether to use quantum execution first
        
    Returns:
        Dictionary with comparison results
    """
    # Import here to avoid circular imports
    from model import forward_pass
    
    # Run with first mode
    mode1_name = "quantum" if use_quantum else "classical"
    start = time.time()
    result1 = forward_pass(model, batch, use_quantum=use_quantum)
    time1 = time.time() - start
    
    # Run with second mode
    mode2_name = "classical" if use_quantum else "quantum"
    start = time.time()
    result2 = forward_pass(model, batch, use_quantum=not use_quantum)
    time2 = time.time() - start
    
    # Calculate difference between results
    if isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
        abs_diff = torch.abs(result1 - result2)
        mean_diff = torch.mean(abs_diff).item()
        max_diff = torch.max(abs_diff).item()
    else:
        mean_diff = "N/A"
        max_diff = "N/A"
    
    return {
        f"{mode1_name}_time": time1,
        f"{mode2_name}_time": time2,
        "speedup": time2 / time1 if time1 > 0 else float('inf'),
        "mean_diff": mean_diff,
        "max_diff": max_diff
    }

def visualize_quantum_circuit(circuit, filename=None):
    """
    Visualize a quantum circuit and optionally save to file
    
    Args:
        circuit: Qiskit QuantumCircuit to visualize
        filename: If provided, save figure to this path
    """
    fig = circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig

def evaluate_circuit_quality(circuit, backend=None):
    """
    Analyze the circuit's quality metrics
    
    Args:
        circuit: Qiskit QuantumCircuit to evaluate
        backend: Backend to use for transpilation (if None, use simulator)
    
    Returns:
        dict: Quality metrics including depth, gate counts, etc.
    """
    # Fix: Update import to use qiskit_aer instead of qiskit.providers.aer
    from qiskit_aer import AerSimulator
    backend = backend or AerSimulator()
    
    # Transpile circuit for the target backend
    transpiled = transpile(circuit, backend)
    
    # Fix: Compute depth directly from the circuit without using the Depth pass
    metrics = {
        'depth': transpiled.depth(),  # Use depth() method directly
        'width': transpiled.num_qubits,
        'gates': transpiled.count_ops(),
        'total_gates': sum(transpiled.count_ops().values())
    }
    
    return metrics

def apply_error_mitigation(counts, noise_level=0.01):
    """
    Simple error mitigation technique for quantum results
    
    Args:
        counts: Results from quantum circuit execution
        noise_level: Estimated noise level to mitigate
    
    Returns:
        dict: Mitigated counts
    """
    # This is a simplified error mitigation technique
    # In a real system, this would be more sophisticated
    mitigated_counts = {}
    total = sum(counts.values())
    
    # Apply a simple thresholding to reduce noise
    for state, count in counts.items():
        if count / total > noise_level:
            mitigated_counts[state] = count
        else:
            # Redistribute counts from noisy results
            if '0' * len(state) in mitigated_counts:
                mitigated_counts['0' * len(state)] += count
            else:
                mitigated_counts['0' * len(state)] = count
    
    # Renormalize
    new_total = sum(mitigated_counts.values())
    for state in mitigated_counts:
        mitigated_counts[state] = int((mitigated_counts[state] / new_total) * total)
    
    return mitigated_counts

def compare_classical_quantum(classical_result, quantum_result, plot=True):
    """
    Compare classical and quantum computation results
    
    Args:
        classical_result: NumPy array of classical computation result
        quantum_result: NumPy array of quantum computation result
        plot: Whether to generate visualization plot
    
    Returns:
        tuple: (fidelity, mse, plot_figure)
    """
    # Calculate mean squared error
    mse = np.mean((classical_result - quantum_result) ** 2)
    
    # Calculate a quasi-fidelity (not true quantum state fidelity)
    classical_norm = np.linalg.norm(classical_result)
    quantum_norm = np.linalg.norm(quantum_result)
    if (classical_norm > 0 and quantum_norm > 0):
        classical_normalized = classical_result / classical_norm
        quantum_normalized = quantum_result / quantum_norm
        fidelity = np.abs(np.dot(classical_normalized, quantum_normalized)) ** 2
    else:
        fidelity = 0
    
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Limit to first 10 elements for clarity if larger
        display_size = min(10, len(classical_result))
        indices = np.arange(display_size)
        
        bar_width = 0.35
        ax.bar(indices - bar_width/2, classical_result[:display_size], 
               bar_width, label='Classical', alpha=0.7)
        ax.bar(indices + bar_width/2, quantum_result[:display_size], 
               bar_width, label='Quantum', alpha=0.7)
        
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Value')
        ax.set_title('Classical vs Quantum Result Comparison')
        ax.set_xticks(indices)
        ax.legend()
        plt.tight_layout()
    
    return {
        'fidelity': fidelity,
        'mse': mse,
        'plot': fig
    }

def quantum_resource_estimator(circuit):
    """
    Estimate quantum resources required for this circuit
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        dict: Resource estimates (gates, depth, qubits)
    """
    num_qubits = circuit.num_qubits
    # Fix: Get depth directly from the circuit
    depth = circuit.depth()
    gate_counts = circuit.count_ops()
    
    # Estimate T gates (important for quantum resource estimation)
    t_count = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
    
    # Estimate CNOT gates (important for error rates)
    cx_count = gate_counts.get('cx', 0)
    
    return {
        'qubits': num_qubits,
        'depth': depth,
        'gate_counts': gate_counts,
        't_count': t_count,
        'cx_count': cx_count,
        'estimated_error_rate': 0.01 * cx_count  # Simplified model assuming 1% error per CNOT
    }

def setup_sampler_from_config(config):
    """
    Create appropriate Sampler primitive from a config object.
    
    This is a utility function to help transition from QuantumInstance to primitives
    
    Args:
        config: Configuration object with backend and run parameters
        
    Returns:
        Configured Sampler primitive
    """
    backend = config.backend
    
    # Determine primitive type based on backend
    if hasattr(backend, 'configuration'):
        # Create BackendSampler for a real backend
        # Fix: Update to use the correct BackendSampler API
        options = {
            "optimization_level": config.circuit_optimization_level if hasattr(config, 'circuit_optimization_level') else 1,
            "shots": config.shots if hasattr(config, 'shots') else 1024
        }
        
        # For Aer backend, add noise model if provided
        if hasattr(config, 'noise_model') and config.noise_model is not None:
            options["noise_model"] = config.noise_model
        
        return BackendSampler(
            backend=backend,
            options=options
        )
    else:
        # Use reference Sampler for simulator
        return Sampler()
