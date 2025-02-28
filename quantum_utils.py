import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity, Statevector
import logging

logger = logging.getLogger('HybrIQ.quantum_utils')

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
    from qiskit.providers.aer import AerSimulator
    backend = backend or AerSimulator()
    
    # Transpile circuit for the target backend
    transpiled = transpile(circuit, backend)
    
    # Compute metrics
    metrics = {
        'depth': transpiled.depth(),
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
    if classical_norm > 0 and quantum_norm > 0:
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
