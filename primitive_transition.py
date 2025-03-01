"""
Utility module to help transition from QuantumInstance to primitives in HybrIQ.

This module provides compatibility wrappers for code designed to work with 
QuantumInstance to help it work with the new primitives approach.
"""

from qiskit import QuantumCircuit
from qiskit.primitives import BackendSampler, Sampler, BackendEstimator, Estimator
from qiskit_aer import Aer
import logging

logger = logging.getLogger('HybrIQ.transition')

class QuantumInstanceWrapper:
    """
    A wrapper to make primitives behave like a QuantumInstance for backward compatibility.
    
    This class provides a simplified interface to help transition code from QuantumInstance
    to primitives-based execution.
    """
    
    def __init__(self, backend=None, shots=1024, optimization_level=1, 
                 noise_model=None, measurement_error_mitigation=False):
        """
        Initialize a QuantumInstance-like wrapper around primitives.
        
        Args:
            backend (Backend): Backend to execute circuits on
            shots (int): Number of shots to use
            optimization_level (int): Transpiler optimization level
            noise_model (NoiseModel): Noise model to apply for simulation
            measurement_error_mitigation (bool): Whether to apply error mitigation
        """
        self.backend = backend if backend is not None else Aer.get_backend('qasm_simulator')
        self.shots = shots
        self.optimization_level = optimization_level
        self.noise_model = noise_model
        self.measurement_error_mitigation = measurement_error_mitigation
        
        # Create appropriate sampler
        self.sampler = self._create_sampler()
        
        # Create appropriate estimator
        self.estimator = self._create_estimator()
        
        logger.info(f"Created QuantumInstance wrapper with backend: {self.backend.name}")
    
    def _create_sampler(self):
        """Create appropriate sampler based on the backend"""
        # Fix: Update to use the correct BackendSampler API
        options = {
            "optimization_level": self.optimization_level,
            "shots": self.shots
        }
        
        # For Aer backend, add noise model if provided
        if hasattr(self, 'noise_model') and self.noise_model is not None:
            options["noise_model"] = self.noise_model
            
        # Create sampler
        return BackendSampler(
            backend=self.backend,
            options=options
        )
    
    def _create_estimator(self):
        """Create appropriate estimator based on the backend"""
        # Fix: Update to use the correct BackendEstimator API
        options = {
            "optimization_level": self.optimization_level,
            "shots": self.shots
        }
        
        # For Aer backend, add noise model if provided
        if hasattr(self, 'noise_model') and self.noise_model is not None:
            options["noise_model"] = self.noise_model
            
        # Create estimator
        return BackendEstimator(
            backend=self.backend,
            options=options
        )
    
    def execute(self, circuits):
        """
        Execute the given circuits.
        
        Args:
            circuits: A single or list of QuantumCircuit objects to execute
            
        Returns:
            A result object with modified interface to match QuantumInstance results
        """
        if not isinstance(circuits, list):
            circuits = [circuits]
        
        # Use sampler to run the circuits
        sampler_result = self.sampler.run(circuits).result()
        
        # Wrap the result to provide compatibility
        return PrimitivesResultWrapper(sampler_result)
    
    def transpile(self, circuits):
        """
        Transpile the given circuits for the backend
        
        Args:
            circuits: A single or list of QuantumCircuit objects to transpile
            
        Returns:
            Transpiled circuits
        """
        from qiskit import transpile
        return transpile(
            circuits, 
            self.backend, 
            optimization_level=self.optimization_level,
        )

class PrimitivesResultWrapper:
    """
    A wrapper to make primitive results compatible with QuantumInstance results
    """
    
    def __init__(self, primitive_result):
        """
        Create a wrapper around a primitive result
        
        Args:
            primitive_result: The SamplerResult to wrap
        """
        self.primitive_result = primitive_result
        self.results = self._create_results()
    
    def _create_results(self):
        """Create results list in QuantumInstance style"""
        results = []
        
        # Convert each quasi-distribution to a result object
        for i, quasi_dist in enumerate(self.primitive_result.quasi_dists):
            # Create result data with counts
            counts = {}
            for state, prob in quasi_dist.items():
                # Only include states with positive probability
                if prob > 0:
                    # Convert to binary string
                    bitstring = bin(state)[2:].zfill(len(bin(max(quasi_dist.keys(), default=0))[2:]))
                    counts[bitstring] = int(prob * self.primitive_result.metadata[i].get("shots", 1024))
            
            # Create a simplified result object
            result = SimpleResult(counts)
            results.append(result)
        
        return results
    
    def get_counts(self, idx=0):
        """Get counts for the specified circuit"""
        if idx < len(self.results):
            return self.results[idx].data.counts
        return {}

class SimpleResult:
    """A simplified result class that mimics a QuantumInstance result"""
    
    class Data:
        def __init__(self, counts):
            self.counts = counts
    
    def __init__(self, counts):
        self.data = self.Data(counts)

def get_primitive_from_quantum_instance(quantum_instance):
    """
    Convert a QuantumInstance to appropriate primitives
    
    Args:
        quantum_instance: A QuantumInstance object
        
    Returns:
        tuple: (sampler, estimator) primitives
    """
    # Extract parameters from quantum_instance
    backend = quantum_instance.backend
    shots = quantum_instance.run_config.shots
    optimization_level = quantum_instance.run_config.optimization_level
    noise_model = getattr(quantum_instance, 'noise_model', None)
    
    # Fix: Update to use the correct BackendSampler API
    options = {
        "optimization_level": optimization_level,
        "shots": shots
    }
    
    # For Aer backend, add noise model if provided
    if noise_model is not None:
        options["noise_model"] = noise_model
    
    # Create primitives
    sampler = BackendSampler(
        backend=backend,
        options=options
    )
    
    estimator = BackendEstimator(
        backend=backend,
        options=options
    )
    
    return sampler, estimator
