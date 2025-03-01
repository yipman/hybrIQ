"""
Utility functions for quantum matrix operations.
"""

import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger('HybrIQ.matrix_utils')

def create_unitary_from_matrix(matrix, method='svd'):
    """
    Convert an arbitrary matrix to a unitary matrix.
    
    Args:
        matrix (np.ndarray): Input matrix
        method (str): Method to use for creating unitary:
                     'svd': Singular value decomposition (U @ V†)
                     'polar': Polar decomposition
                     'gram': Gram-Schmidt process
        
    Returns:
        np.ndarray: A unitary matrix approximating the input matrix
    """
    # Ensure matrix is square
    n = min(matrix.shape)
    resized_matrix = matrix[:n, :n]
    
    if method == 'svd':
        # Use SVD decomposition (U @ V†)
        try:
            u, _, vh = np.linalg.svd(resized_matrix, full_matrices=True)
            unitary = u @ vh
            
            # Double-check unitarity
            if not is_unitary(unitary, atol=1e-10):
                logger.warning("SVD result is not perfectly unitary, applying correction")
                u, _, vh = np.linalg.svd(unitary, full_matrices=True)
                unitary = u @ vh
                
            return unitary
                
        except Exception as e:
            logger.error(f"SVD decomposition failed: {str(e)}")
            return np.eye(n)
            
    elif method == 'polar':
        # Use polar decomposition
        try:
            unitary, _ = linalg.polar(resized_matrix)
            return unitary
        except Exception as e:
            logger.error(f"Polar decomposition failed: {str(e)}")
            return np.eye(n)
            
    elif method == 'gram':
        # Use Gram-Schmidt process
        try:
            q, r = np.linalg.qr(resized_matrix)
            return q
        except Exception as e:
            logger.error(f"QR decomposition failed: {str(e)}")
            return np.eye(n)
            
    else:
        logger.error(f"Unknown unitary creation method: {method}")
        return np.eye(n)

def is_unitary(matrix, atol=1e-8):
    """
    Check if a matrix is unitary within specified tolerance.
    
    Args:
        matrix (np.ndarray): Matrix to check
        atol (float): Absolute tolerance for equality check
        
    Returns:
        bool: True if the matrix is unitary
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
        
    n = matrix.shape[0]
    identity = np.eye(n)
    
    # Test U @ U† = I
    product = matrix @ matrix.conj().T
    return np.allclose(product, identity, atol=atol)

def get_matrix_properties(matrix):
    """
    Analyze properties of a matrix.
    
    Args:
        matrix (np.ndarray): Matrix to analyze
        
    Returns:
        dict: Dictionary of matrix properties
    """
    # Basic properties
    shape = matrix.shape
    rank = np.linalg.matrix_rank(matrix)
    
    # Check if matrix is square
    is_square = shape[0] == shape[1]
    
    # Only calculate these if matrix is square
    properties = {
        'shape': shape,
        'rank': rank,
        'is_square': is_square,
        'norm': np.linalg.norm(matrix)
    }
    
    if is_square:
        # Calculate determinant, eigenvalues
        try:
            properties['determinant'] = np.linalg.det(matrix)
            properties['eigenvalues'] = np.linalg.eigvals(matrix)
            properties['is_unitary'] = is_unitary(matrix)
            properties['is_hermitian'] = np.allclose(matrix, matrix.conj().T)
            properties['is_normal'] = np.allclose(matrix @ matrix.conj().T, 
                                                matrix.conj().T @ matrix)
        except Exception as e:
            logger.error(f"Error calculating matrix properties: {str(e)}")
    
    return properties

def decompose_for_quantum(matrix, max_qubits=None):
    """
    Decompose a matrix for quantum circuit processing.
    
    Args:
        matrix (np.ndarray): Matrix to decompose
        max_qubits (int, optional): Maximum number of qubits to use
        
    Returns:
        dict: Decomposition data including unitary transformation
    """
    # Determine number of qubits needed
    n = matrix.shape[0]
    num_qubits = int(np.ceil(np.log2(n)))
    
    # Limit qubits if specified
    if max_qubits is not None:
        num_qubits = min(num_qubits, max_qubits)
        n = 2**num_qubits
        matrix = matrix[:n, :n]
    
    # Pad to power of 2 if needed
    padded_size = 2**num_qubits
    if n < padded_size:
        padded_matrix = np.zeros((padded_size, padded_size), dtype=complex)
        padded_matrix[:n, :n] = matrix
        matrix = padded_matrix
    
    # Create unitary using SVD
    unitary = create_unitary_from_matrix(matrix)
    
    # Get singular values to understand scaling
    _, s, _ = np.linalg.svd(matrix)
    
    return {
        'num_qubits': num_qubits,
        'unitary': unitary,
        'singular_values': s,
        'scale_factor': np.mean(s),
        'original_matrix': matrix
    }
