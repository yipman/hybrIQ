"""
Circuit optimization utilities for HybrIQ.
Provides optimized quantum circuit compilation using Qiskit transpiler passes.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CXCancellation,
    CommutativeCancellation,
    Collect2qBlocks,
    ConsolidateBlocks,
    Depth,
    FixedPoint,
    RemoveBarriers,
    Unroll3qOrMore,
    SabreLayout,
    SabreSwap,
    BasicSwap,
    StochasticSwap,
    CSPLayout,
    DenseLayout
)
# Import TrivialLayout from the correct location for Qiskit 1.0
from qiskit.transpiler.passes import TrivialLayout
import logging

logger = logging.getLogger('HybrIQ')

def get_optimized_layout_pass_manager(backend, optimization_level=3):
    """
    Create a pass manager for optimizing qubit layout.
    
    Args:
        backend: Quantum backend to optimize for
        optimization_level: Level of optimization (1-3)
        
    Returns:
        PassManager configured for layout optimization
    """
    if optimization_level == 1:
        # Simple layout strategy
        layout_method = TrivialLayout(backend.configuration().coupling_map)
    elif optimization_level == 2:
        # Balanced approach
        layout_method = DenseLayout(backend.configuration().coupling_map)
    else:  # optimization_level >= 3
        # Most thorough but slowest approach
        layout_method = SabreLayout(backend.configuration().coupling_map, 
                                   max_iterations=10,
                                   swap_trials=20)
    
    pm = PassManager()
    pm.append(layout_method)
    return pm

def get_optimized_routing_pass_manager(backend, optimization_level=3):
    """
    Create a pass manager for optimizing qubit routing.
    
    Args:
        backend: Quantum backend to optimize for
        optimization_level: Level of optimization (1-3)
        
    Returns:
        PassManager configured for routing optimization
    """
    coupling_map = backend.configuration().coupling_map
    
    pm = PassManager()
    if optimization_level == 1:
        # Faster but less optimal
        pm.append(BasicSwap(coupling_map))
    elif optimization_level == 2:
        # Medium complexity
        pm.append(StochasticSwap(coupling_map))
    else:  # optimization_level >= 3
        # Most thorough but slowest approach
        pm.append(SabreSwap(coupling_map, heuristic='decay', seed=42))
    
    return pm

def get_optimized_gate_cancellation_pass_manager(optimization_level=2):
    """
    Create a pass manager for gate cancellation and simplification.
    
    Args:
        optimization_level: Level of optimization (1-3)
        
    Returns:
        PassManager configured for gate optimization
    """
    pm = PassManager()
    
    # Basic gate optimizations
    pm.append(RemoveBarriers())
    pm.append(Optimize1qGates())
    pm.append(CXCancellation())
    
    if optimization_level >= 2:
        # Medium optimizations
        pm.append(CommutativeCancellation())
    
    if optimization_level >= 3:
        # Advanced optimizations
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        # Run single-qubit optimizations again after consolidation
        pm.append(Optimize1qGates())
    
    # Run until no more optimizations can be applied
    pm.append(FixedPoint('size'))
    
    return pm

def optimize_circuit(circuit, backend, optimization_level=3):
    """
    Optimize a quantum circuit for the given backend.
    
    Args:
        circuit: QuantumCircuit to optimize
        backend: Backend to target
        optimization_level: Level of optimization (1-3)
    
    Returns:
        Optimized QuantumCircuit
    """
    # Fix: Get circuit depth directly without using the recurse parameter
    start_depth = circuit.depth()
    start_size = circuit.size()
    
    # Use Qiskit's built-in transpile with our optimization level
    optimized_circuit = transpile(circuit, 
                                 backend=backend, 
                                 optimization_level=optimization_level,
                                 seed_transpiler=42)
    
    # Additional custom optimizations
    gate_pm = get_optimized_gate_cancellation_pass_manager(optimization_level)
    optimized_circuit = gate_pm.run(optimized_circuit)
    
    # Fix: Get circuit depth directly without using the recurse parameter
    end_depth = optimized_circuit.depth()
    end_size = optimized_circuit.size()
    
    logger.debug(f"Circuit optimization: depth {start_depth}->{end_depth}, "
                f"size {start_size}->{end_size}")
    
    return optimized_circuit

def optimize_inner_product_circuit(circuit, backend, optimization_level=3):
    """
    Optimize a quantum circuit specifically for inner product calculations.
    
    Args:
        circuit: QuantumCircuit for inner product
        backend: Backend to target
        optimization_level: Level of optimization (1-3)
    
    Returns:
        Optimized QuantumCircuit
    """
    # Inner product circuits are typically simple, so we focus on gate optimizations
    return optimize_circuit(circuit, backend, optimization_level)

def optimize_matrix_vector_circuit(circuit, backend, optimization_level=3):
    """
    Optimize a quantum circuit specifically for matrix-vector multiplication.
    
    Args:
        circuit: QuantumCircuit for matrix-vector multiplication
        backend: Backend to target
        optimization_level: Level of optimization (1-3)
    
    Returns:
        Optimized QuantumCircuit
    """
    # Matrix-vector circuits can be complex, so we apply more aggressive optimizations
    optimized = optimize_circuit(circuit, backend, optimization_level)
    
    # For matrix operations, consolidating blocks can be particularly effective
    if optimization_level >= 2:
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Optimize1qGates())
        optimized = pm.run(optimized)
    
    return optimized
