"""
Feedforward layer implementations for HybrIQ.
"""

import numpy as np
import torch
from functools import lru_cache
import logging

logger = logging.getLogger('HybrIQ.feedforward')

def gelu(x):
    """GELU activation function."""
    return x * torch.sigmoid(1.702 * x)

class ClassicalFeedForward:
    """Classical implementation of feedforward layer."""
    
    def __init__(self, d_model, d_ff):
        """
        Initialize a classical feedforward layer.
        
        Args:
            d_model (int): Input/output dimension
            d_ff (int): Hidden dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights and biases
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """
        Forward pass through feedforward layer.
        
        Args:
            x (torch.Tensor): Input of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output of same shape as input
        """
        # Convert to numpy for efficient matrix operations
        x_numpy = x.numpy()
        batch_size, seq_len, d_model = x.shape
        
        # First linear layer
        ff_intermediate = np.matmul(x_numpy.reshape(-1, d_model), self.W1)
        ff_intermediate = ff_intermediate.reshape(batch_size, seq_len, -1) + self.b1
        
        # Activation
        ff_intermediate_tensor = torch.tensor(ff_intermediate)
        ff_activated = gelu(ff_intermediate_tensor)
        
        # Second linear layer
        ff_output = ff_activated @ torch.tensor(self.W2) + torch.tensor(self.b2)
        
        return ff_output

class QuantumFeedForward:
    """Quantum-enhanced implementation of feedforward layer."""
    
    def __init__(self, d_model, d_ff, config):
        """
        Initialize a quantum feedforward layer.
        
        Args:
            d_model (int): Input/output dimension
            d_ff (int): Hidden dimension
            config: Configuration object with quantum settings
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.config = config
        
        # Initialize weights and biases
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
        
        # Calculate the number of qubits for quantum computation
        self.num_qubits = min(int(np.ceil(np.log2(max(d_model, d_ff)))), 
                              config.max_circuit_width)
    
    def forward(self, x):
        """
        Forward pass through quantum-enhanced feedforward layer.
        
        Args:
            x (torch.Tensor): Input of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output of same shape as input
        """
        from model import quantum_matrix_vector_mult
        
        # Convert to numpy for quantum processing
        x_numpy = x.numpy()
        batch_size, seq_len, d_model = x.shape
        
        # Use quantum matrix-vector multiplication for the first layer
        if self.config.should_use_quantum(d_model):
            ff_intermediate = np.zeros((batch_size, seq_len, self.d_ff))
            
            # Process each token in the sequence
            for b in range(batch_size):
                for s in range(seq_len):
                    try:
                        # Quantum matrix-vector multiplication for W1
                        ff_intermediate[b, s] = quantum_matrix_vector_mult(
                            self.W1, 
                            x_numpy[b, s], 
                            num_qubits=self.num_qubits
                        )
                    except Exception as e:
                        # Fallback to classical if quantum fails
                        logger.warning(f"Quantum computation failed, using classical fallback: {str(e)}")
                        ff_intermediate[b, s] = self.W1 @ x_numpy[b, s]
        else:
            # Classical computation for larger dimensions
            ff_intermediate = np.matmul(x_numpy.reshape(-1, d_model), self.W1)
            ff_intermediate = ff_intermediate.reshape(batch_size, seq_len, -1)
        
        # Add bias
        ff_intermediate = ff_intermediate + self.b1
        
        # Activation
        ff_intermediate_tensor = torch.tensor(ff_intermediate)
        ff_activated = gelu(ff_intermediate_tensor)
        
        # Second linear layer (always classical for efficiency)
        ff_output = ff_activated @ torch.tensor(self.W2) + torch.tensor(self.b2)
        
        return ff_output

class HybridFeedForward:
    """Hybrid implementation that can switch between classical and quantum."""
    
    def __init__(self, d_model, d_ff, config):
        """
        Initialize a hybrid feedforward layer.
        
        Args:
            d_model (int): Input/output dimension
            d_ff (int): Hidden dimension
            config: Configuration object with settings
        """
        self.config = config
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize both implementations
        self.classical = ClassicalFeedForward(d_model, d_ff)
        self.quantum = QuantumFeedForward(d_model, d_ff, config)
        
        # Share weights between implementations
        self.quantum.W1 = self.classical.W1
        self.quantum.b1 = self.classical.b1
        self.quantum.W2 = self.classical.W2
        self.quantum.b2 = self.classical.b2
        
        # Decide which implementation to use based on input size
        self.use_quantum = config.use_quantum and d_model <= config.quantum_threshold
    
    def forward(self, x):
        """
        Forward pass through the appropriate implementation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.use_quantum:
            return self.quantum.forward(x)
        else:
            return self.classical.forward(x)
