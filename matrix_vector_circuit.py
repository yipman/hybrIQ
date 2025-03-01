"""
Implementation of quantum matrix-vector multiplication.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import logging

# Import utility functions
from quantum_matrix_utils import decompose_for_quantum, create_unitary_from_matrix

logger = logging.getLogger('HybrIQ.matrix_vector')

def create_matrix_vector_circuit(matrix, vector, num_qubits=None, method='basic'):
    """
    Create a quantum circuit for matrix-vector multiplication.
    
    Args:
        matrix (np.ndarray): Matrix for multiplication
        vector (np.ndarray): Vector to multiply
        num_qubits (int, optional): Number of qubits to use
        method (str): Circuit creation method ('basic', 'block_encoding', etc.)
        
    Returns:
        tuple: (quantum_circuit, scale_factor)
    """
    # Determine number of qubits needed
    if num_qubits is None:
        num_qubits = int(np.ceil(np.log2(max(matrix.shape[0], len(vector)))))
    
    # Normalize vector
    vector_norm = np.linalg.norm(vector)
    if vector_norm > 0:
        normalized_vector = vector / vector_norm
    else:
        normalized_vector = np.zeros_like(vector)
    
    # Decompose matrix for quantum processing
    decomposition = decompose_for_quantum(matrix, max_qubits=num_qubits)
    unitary_matrix = decomposition['unitary']
    scale_factor = decomposition['scale_factor']
    
    # Create circuit based on specified method
    if method == 'basic':
        return create_basic_matrix_vector_circuit(unitary_matrix, normalized_vector, num_qubits), scale_factor
    elif method == 'block_encoding':
        return create_block_encoding_circuit(unitary_matrix, normalized_vector, num_qubits), scale_factor
    else:
        logger.warning(f"Unknown circuit method '{method}', using basic")
        return create_basic_matrix_vector_circuit(unitary_matrix, normalized_vector, num_qubits), scale_factor

def create_basic_matrix_vector_circuit(unitary_matrix, vector, num_qubits):
    """
    Create a basic circuit for matrix-vector multiplication.
    
    Args:
        unitary_matrix (np.ndarray): Unitary matrix for multiplication
        vector (np.ndarray): Normalized vector to multiply
        num_qubits (int): Number of qubits to use
        
    Returns:
        QuantumCircuit: Circuit implementing the matrix-vector multiplication
    """
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Pad vector if needed
    padded_vector = np.zeros(2**num_qubits, dtype=complex)
    padded_vector[:len(vector)] = vector
    
    # Initialize with vector
    qc.initialize(padded_vector, range(num_qubits))
    
    # Apply unitary matrix
    qc.unitary(Operator(unitary_matrix), range(num_qubits), label='U')
    
    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc

def create_block_encoding_circuit(unitary_matrix, vector, num_qubits):
    """
    Create a circuit for matrix-vector multiplication using block encoding.
    
    Args:
        unitary_matrix (np.ndarray): Unitary matrix for multiplication
        vector (np.ndarray): Normalized vector to multiply
        num_qubits (int): Number of qubits to use
        
    Returns:
        QuantumCircuit: Circuit implementing the matrix-vector multiplication
    """
    # Add one ancilla qubit for block encoding
    total_qubits = num_qubits + 1
    
    # Create quantum circuit
    qc = QuantumCircuit(total_qubits, num_qubits)
    
    # Pad vector if needed
    padded_vector = np.zeros(2**num_qubits, dtype=complex)
    padded_vector[:len(vector)] = vector
    
    # Initialize data qubits with vector state
    qc.initialize(padded_vector, range(1, total_qubits))
    
    # Put ancilla in superposition
    qc.h(0)
    
    # Create controlled unitary operation
    # We include both the matrix and its conjugate transpose in a block encoding
    block_unitary = np.block([
        [np.eye(2**num_qubits), np.zeros((2**num_qubits, 2**num_qubits))],
        [np.zeros((2**num_qubits, 2**num_qubits)), unitary_matrix]
    ])
    
    # Apply block unitary operation
    qc.unitary(Operator(block_unitary), range(total_qubits), label='Block_U')
    
    # Apply Hadamard to ancilla again
    qc.h(0)
    
    # Measure data qubits
    qc.measure(range(1, total_qubits), range(num_qubits))
    
    return qc
