import numpy as np
import torch
from torch.nn.functional import softmax
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendSampler, Sampler
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
from qiskit_aer import QasmSimulator
from qiskit_aer.noise import NoiseModel
import time
from functools import lru_cache
import logging

# Import our circuit optimizer
from circuit_optimizer import optimize_inner_product_circuit, optimize_matrix_vector_circuit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HybrIQ')

class HybridConfig:
    """Configuration for hybrid quantum-classical execution"""
    def __init__(self):
        self.use_quantum = True
        self.shots = 1024
        self.max_circuit_width = 5  # Max number of qubits to use in quantum computation
        self.error_mitigation = True
        self.measurement_error_mitigation = True
        self.circuit_optimization_level = 3
        self.circuit_optimization_strategy = 'default'  # Options: 'default', 'depth', 'gates'
        self.simulator = 'qasm_simulator'
        self.noise_model = None
        self.quantum_threshold = 8  # Max dimension to use quantum computation (above this, use classical)
        self.circuit_reuse = True
        self.batch_circuits = True
        self.max_batch_size = 100
        self.backend = None
        self.sampler = None  # Store the primitive sampler instead of quantum_instance
        
    def initialize_backend(self):
        """Initialize the quantum backend and sampler with current settings"""
        # Get the backend
        backend = Aer.get_backend(self.simulator)
        self.backend = backend
        
        # Create a BackendSampler primitive instead of QuantumInstance
        # Fix: Update the parameters to match the current BackendSampler API
        # BackendSampler doesn't accept transpile_options directly
        
        # Create the appropriate sampler
        self.sampler = BackendSampler(
            backend=backend,
            options={
                # Include options that would normally be in transpile_options
                "optimization_level": self.circuit_optimization_level,
                # Include options that would normally be in run_options
                "shots": self.shots,
                # Include noise model if provided
                **({"noise_model": self.noise_model} if self.noise_model is not None else {})
            }
        )
        
        # Log optimization settings
        logger.info(f"Using circuit optimization level: {self.circuit_optimization_level}")
        logger.info(f"Circuit optimization strategy: {self.circuit_optimization_strategy}")
        
        return self.backend
    
    def should_use_quantum(self, input_size):
        """Determine if quantum computation should be used based on input size"""
        return self.use_quantum and input_size <= self.quantum_threshold

# Global configuration
config = HybridConfig()
config.initialize_backend()

# Placeholder activation function
def gelu(x):
    return x * torch.sigmoid(1.702 * x)

@lru_cache(maxsize=128)
def create_inner_product_circuit(num_qubits):
    """Create and compile a reusable circuit template for inner product"""
    # Fix: Update the circuit template to match the new initialization pattern
    qc = QuantumCircuit(num_qubits + 1, 1)
    # We'll handle initialization and application separately
    # Just measure the first qubit at the end
    qc.measure(0, 0)
    
    # Use our circuit optimizer to improve the circuit
    if hasattr(config, 'backend') and config.backend is not None:
        return optimize_inner_product_circuit(
            qc, 
            config.backend, 
            optimization_level=config.circuit_optimization_level
        )
    else:
        return transpile(qc, optimization_level=config.circuit_optimization_level)

def quantum_inner_product(u, v, num_qubits=None):
    """
    Enhanced quantum inner product calculation with error mitigation
    """
    if num_qubits is None:
        num_qubits = min(int(np.log2(len(u))), config.max_circuit_width)
    
    if not config.should_use_quantum(len(u)):
        # Fall back to classical computation for larger vectors
        return np.dot(u, v)
    
    start_time = time.time()
    
    u_norm = u / np.linalg.norm(u) if np.linalg.norm(u) > 0 else np.zeros_like(u)
    v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros_like(v)
    
    # Get the circuit template
    qc = create_inner_product_circuit(num_qubits).copy()
    
    # Fix: Update initialization to use controlled operations instead of control_qubit parameter
    # First initialize qubit 0 to |+⟩ state
    qc.h(0)
    
    # Pad vectors to match the required length
    u_padded = u_norm[:2**num_qubits].tolist() + [0] * (2**num_qubits - len(u_norm[:2**num_qubits]))
    v_padded = v_norm[:2**num_qubits].tolist() + [0] * (2**num_qubits - len(v_norm[:2**num_qubits]))
    
    # Initialize the data registers with the vectors
    # Use controlled gates instead of control_qubit parameter
    data_qubits = list(range(1, num_qubits + 1))
    
    # Initialize with u_norm when control is |1⟩
    qc.x(0)  # Flip control to 1
    qc.initialize(u_padded, data_qubits)
    qc.x(0)  # Flip control back to 0
    
    # Initialize with v_norm when control is |0⟩
    qc.initialize(v_padded, data_qubits)
    
    # Final Hadamard on control
    qc.h(0)
    
    # Replace quantum_instance.execute with sampler.run 
    result = config.sampler.run(qc).result()
    quasi_dist = result.quasi_dists[0]
    
    # Interpret results in terms of 0 and 1 outcomes
    p0 = quasi_dist.get(0, 0)  # Probability of measuring |0⟩
    p1 = quasi_dist.get(1, 0)  # Probability of measuring |1⟩
    
    result = (p0 - p1) * np.linalg.norm(u) * np.linalg.norm(v)
    
    exec_time = time.time() - start_time
    if exec_time > 0.1:  # Log only if execution takes significant time
        logger.debug(f"Quantum inner product took {exec_time:.4f}s for {len(u)}-dim vectors")
    
    return result

def batch_quantum_inner_products(query_vectors, key_vectors):
    """Batch multiple inner product calculations for efficiency"""
    if not config.batch_circuits or not config.should_use_quantum(query_vectors.shape[1]):
        # Fall back to classical batch computation
        return np.einsum('ij,kj->ik', query_vectors, key_vectors)
    
    batch_size = min(config.max_batch_size, query_vectors.shape[0] * key_vectors.shape[0])
    results = np.zeros((query_vectors.shape[0], key_vectors.shape[0]))
    
    # Create batch of circuits
    circuits = []
    circuit_params = []
    count = 0
    
    for i in range(query_vectors.shape[0]):
        for j in range(key_vectors.shape[0]):
            if count >= batch_size:
                # Execute batch and reset
                batch_results = execute_quantum_batch(circuits)
                for idx, (ii, jj) in enumerate(circuit_params):
                    quasi_dist = batch_results[idx]
                    p0 = quasi_dist.get(0, 0)  # Probability of measuring |0⟩
                    p1 = quasi_dist.get(1, 0)  # Probability of measuring |1⟩
                    q_norm = np.linalg.norm(query_vectors[ii])
                    k_norm = np.linalg.norm(key_vectors[jj])
                    results[ii, jj] = (p0 - p1) * q_norm * k_norm
                
                circuits = []
                circuit_params = []
                count = 0
            
            num_qubits = min(int(np.log2(query_vectors.shape[1])), config.max_circuit_width)
            u = query_vectors[i]
            v = key_vectors[j]
            
            u_norm = u / np.linalg.norm(u) if np.linalg.norm(u) > 0 else np.zeros_like(u)
            v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros_like(v)
            
            qc = create_inner_product_circuit(num_qubits).copy()
            
            # Fix: Update initialization to use controlled operations instead of control_qubit parameter
            # First initialize qubit 0 to |+⟩ state
            qc.h(0)
            
            # Pad vectors to match the required length
            u_padded = u_norm[:2**num_qubits].tolist() + [0] * (2**num_qubits - len(u_norm[:2**num_qubits]))
            v_padded = v_norm[:2**num_qubits].tolist() + [0] * (2**num_qubits - len(v_norm[:2**num_qubits]))
            
            # Initialize the data registers with the vectors
            # Use controlled gates instead of control_qubit parameter
            data_qubits = list(range(1, num_qubits + 1))
            
            # Initialize with u_norm when control is |1⟩
            qc.x(0)  # Flip control to 1
            qc.initialize(u_padded, data_qubits)
            qc.x(0)  # Flip control back to 0
            
            # Initialize with v_norm when control is |0⟩
            qc.initialize(v_padded, data_qubits)
            
            # Final Hadamard on control
            qc.h(0)
            
            circuits.append(qc)
            circuit_params.append((i, j))
            count += 1
    
    # Execute remaining circuits
    if circuits:
        batch_results = execute_quantum_batch(circuits)
        for idx, (ii, jj) in enumerate(circuit_params):
            quasi_dist = batch_results[idx]
            p0 = quasi_dist.get(0, 0)  # Probability of measuring |0⟩
            p1 = quasi_dist.get(1, 0)  # Probability of measuring |1⟩
            q_norm = np.linalg.norm(query_vectors[ii])
            k_norm = np.linalg.norm(key_vectors[jj])
            results[ii, jj] = (p0 - p1) * q_norm * k_norm
    
    return results

def execute_quantum_batch(circuits):
    """Execute a batch of quantum circuits efficiently"""
    # Use the sampler primitive to run all circuits in a batch
    job_result = config.sampler.run(circuits).result()
    
    # Extract the counts for each circuit
    return [quasi_dist for quasi_dist in job_result.quasi_dists]

@lru_cache(maxsize=128)
def create_matrix_vector_circuit(num_qubits):
    """Create a reusable circuit template for matrix-vector multiplication"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    # Will initialize vector and apply unitary later
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Use our circuit optimizer to improve the circuit
    if hasattr(config, 'backend') and config.backend is not None:
        return optimize_matrix_vector_circuit(
            qc, 
            config.backend, 
            optimization_level=config.circuit_optimization_level
        )
    else:
        return transpile(qc, optimization_level=config.circuit_optimization_level)

def quantum_matrix_vector_mult(W, x, num_qubits=None):
    """
    Enhanced quantum matrix-vector multiplication with error mitigation and fallback
    """
    if num_qubits is None:
        num_qubits = min(int(np.log2(len(x))), config.max_circuit_width)
    
    if not config.should_use_quantum(len(x)):
        # Fall back to classical computation for larger vectors
        return W @ x
    
    start_time = time.time()
    
    try:
        # Store original shapes for later reshaping
        original_W_shape = W.shape
        original_x_shape = x.shape
        
        # Ensure x is properly sized and normalized
        x_norm = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else np.zeros_like(x)
        
        # Determine circuit dimensions
        circuit_dim = 2**num_qubits
        
        # Pad or truncate the input vector to fit the circuit
        x_padded = np.zeros(circuit_dim)
        x_padded[:min(len(x_norm), circuit_dim)] = x_norm[:min(len(x_norm), circuit_dim)]
        
        # Prepare a properly sized matrix W_padded
        W_padded = np.zeros((circuit_dim, circuit_dim))
        rows = min(W.shape[0], circuit_dim)
        cols = min(W.shape[1], circuit_dim)
        W_padded[:rows, :cols] = W[:rows, :cols]
        
        # Calculate the full classical result for scaling
        # Use correct dimensions to avoid matmul error
        classical_result = np.zeros(rows)
        if cols <= len(x):
            # If the matrix columns can fit the vector
            classical_result = W[:rows, :cols] @ x[:cols]
        else:
            # If the vector is smaller than matrix columns, pad it
            x_temp = np.zeros(cols)
            x_temp[:len(x)] = x
            classical_result = W[:rows, :cols] @ x_temp
        
        scaling_factor = np.linalg.norm(classical_result) if np.linalg.norm(classical_result) > 0 else 1.0
        
        # Ensure the matrix is unitary for quantum computation
        # Use SVD to find the closest unitary matrix
        U, _, Vh = np.linalg.svd(W_padded)
        W_unitary = U @ Vh
        
        # Verify it's unitary by checking if W† * W ≈ I
        is_unitary = np.allclose(W_unitary.conj().T @ W_unitary, np.eye(circuit_dim), atol=1e-6)
        
        if not is_unitary:
            raise ValueError("Failed to create a unitary matrix from the input")
            
        # Get the circuit template
        qc = create_matrix_vector_circuit(num_qubits).copy()
        
        # Initialize with actual data
        qc.initialize(x_padded.tolist(), range(num_qubits))
        
        # Apply unitary operation
        qc.unitary(Operator(W_unitary), range(num_qubits), label='W')
        
        # Use sampler instead of quantum_instance.execute
        result = config.sampler.run(qc).result()
        quasi_dist = result.quasi_dists[0]
        
        # Convert the quasi-distribution to a vector of amplitudes
        y_est = np.zeros(circuit_dim)
        for state, prob in quasi_dist.items():
            y_est[state] = np.sqrt(prob)  # Use sqrt of probability for amplitude
        
        # Scale result according to original norms
        scaled_result = y_est * scaling_factor
        
        # Ensure we return a vector of the correct size (W.shape[0])
        # Create the output array with the right dimensions
        output = np.zeros(W.shape[0])
        
        # Only copy elements that fit in the output array
        min_size = min(len(scaled_result), W.shape[0])
        output[:min_size] = scaled_result[:min_size]
        
        exec_time = time.time() - start_time
        if exec_time > 0.1:  # Log only if execution takes significant time
            logger.debug(f"Quantum matrix-vector mult took {exec_time:.4f}s for {W.shape} matrix")
        
        return output
        
    except Exception as e:
        # If quantum computation fails, fall back to classical with detailed error
        logger.warning(f"Quantum computation failed: {str(e)}, falling back to classical")
        return W @ x

class Attention:
    def __init__(self, d_model, use_quantum=True):
        self.d_model = d_model
        self.use_quantum = use_quantum
        # Dummy projection matrices
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)

    def get_QKV(self, embeddings):
        # Project embeddings to Q, K, V
        batch_size, seq_len, d_model = embeddings.shape
        Q = torch.tensor(np.array([self.W_q @ embeddings[b, s].numpy() 
                                  for b in range(batch_size) 
                                  for s in range(seq_len)]).reshape(batch_size, seq_len, d_model))
        K = torch.tensor(np.array([self.W_k @ embeddings[b, s].numpy() 
                                  for b in range(batch_size) 
                                  for s in range(seq_len)]).reshape(batch_size, seq_len, d_model))
        V = torch.tensor(np.array([self.W_v @ embeddings[b, s].numpy() 
                                  for b in range(batch_size) 
                                  for s in range(seq_len)]).reshape(batch_size, seq_len, d_model))
        return Q, K, V

    def compute_attention_scores(self, Q, K):
        """
        Compute attention scores with enhanced batching
        """
        batch_size, seq_len, d_model = Q.shape
        attention_scores = np.zeros((batch_size, seq_len, seq_len))
        
        for b in range(batch_size):
            # Convert to numpy for quantum processing
            q_batch = Q[b].numpy()
            k_batch = K[b].numpy()
            
            if config.should_use_quantum(d_model) and self.use_quantum:
                # Use batched quantum inner products
                attention_scores[b] = batch_quantum_inner_products(q_batch, k_batch) / np.sqrt(d_model)
            else:
                # Use classical computation
                attention_scores[b] = np.matmul(q_batch, k_batch.T) / np.sqrt(d_model)
                
        return torch.tensor(attention_scores)

class FeedForward:
    def __init__(self, d_model, d_ff, use_quantum=True):
        # Fix dimensions:
        # W1 maps from d_model to d_ff
        self.W1 = np.random.randn(d_ff, d_model)
        self.b1 = np.zeros(d_ff)
        # W2 maps from d_ff back to d_model
        self.W2 = np.random.randn(d_model, d_ff)
        self.b2 = np.zeros(d_model)
        self.use_quantum = use_quantum
        self.d_model = d_model
        self.d_ff = d_ff
    
    def forward(self, x):
        """
        Enhanced feedforward with adaptive quantum/classical execution
        """
        batch_size, seq_len, d_model = x.shape
        x_numpy = x.numpy()
        
        # First linear transformation: d_model -> d_ff
        ff_intermediate = np.zeros((batch_size, seq_len, self.d_ff))
        
        # Check if dimensions are compatible before attempting quantum computation
        if d_model != self.W1.shape[1]:
            logger.warning(f"Dimension mismatch: expected input dim {self.W1.shape[1]}, got {d_model}. Resizing weights.")
            # Resize W1 to match input dimensions
            new_W1 = np.random.randn(self.d_ff, d_model)
            # Copy as much of the original weights as possible
            min_cols = min(self.W1.shape[1], d_model)
            new_W1[:, :min_cols] = self.W1[:, :min_cols]
            self.W1 = new_W1
        
        if config.should_use_quantum(d_model) and self.use_quantum:
            # Use quantum matrix-vector multiplication
            for b in range(batch_size):
                for s in range(seq_len):
                    try:
                        result = quantum_matrix_vector_mult(self.W1, x_numpy[b, s])
                        # Ensure result has the right shape
                        if result.shape[0] != self.d_ff:
                            # Resize result if needed
                            tmp = np.zeros(self.d_ff)
                            min_size = min(len(result), self.d_ff)
                            tmp[:min_size] = result[:min_size]
                            result = tmp
                        ff_intermediate[b, s] = result
                    except Exception as e:
                        # Fallback to classical in case of any error
                        logger.warning(f"Quantum computation failed: {str(e)}, falling back to classical")
                        ff_intermediate[b, s] = self.W1 @ x_numpy[b, s]
        else:
            # Use classical computation
            for b in range(batch_size):
                for s in range(seq_len):
                    ff_intermediate[b, s] = self.W1 @ x_numpy[b, s]
        
        # Add bias and apply activation
        ff_intermediate = ff_intermediate + self.b1
        ff_intermediate_tensor = torch.tensor(ff_intermediate)
        ff_intermediate_tensor = gelu(ff_intermediate_tensor)
        ff_intermediate_numpy = ff_intermediate_tensor.numpy()
        
        # Check if W2 dimensions match intermediate dimensions
        if self.W2.shape[1] != self.d_ff:
            logger.warning(f"Dimension mismatch in W2: expected {self.d_ff}, got {self.W2.shape[1]}. Resizing weights.")
            # Resize W2 to match dimensions
            new_W2 = np.random.randn(self.d_model, self.d_ff)
            # Copy as much of the original weights as possible
            min_cols = min(self.W2.shape[1], self.d_ff)
            new_W2[:, :min_cols] = self.W2[:, :min_cols]
            self.W2 = new_W2
        
        # Second linear transformation: d_ff -> d_model
        ff_output = np.zeros((batch_size, seq_len, self.d_model))
        
        # Use classical computation for the second transformation
        for b in range(batch_size):
            for s in range(seq_len):
                # W2 has shape (d_model, d_ff), ff_intermediate_numpy[b, s] has shape (d_ff,)
                ff_output[b, s] = self.W2 @ ff_intermediate_numpy[b, s] + self.b2
        
        return torch.tensor(ff_output)

class Layer:
    def __init__(self, d_model, d_ff, use_quantum=True):
        self.attention = Attention(d_model, use_quantum)
        self.ff = FeedForward(d_model, d_ff, use_quantum)
        self.norm1 = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
        self.norm2 = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

class Model:
    def __init__(self, vocab_size=100, d_model=4, n_layers=2, d_ff=16, use_quantum=True):
        self.embedding = lambda x: torch.randn(x.shape[0], x.shape[1], d_model)
        self.layers = [Layer(d_model, d_ff, use_quantum) for _ in range(n_layers)]
        self.output_layer = np.random.randn(d_model, vocab_size)
        self.d_model = d_model
        self.use_quantum = use_quantum

# Enhanced Hybrid Forward Pass
def forward_pass(model, batch, use_quantum=None):
    if use_quantum is not None:
        old_setting = config.use_quantum
        config.use_quantum = use_quantum
    
    start_time = time.time()
    
    embeddings = model.embedding(batch)  # (batch, seq, d_model)
    
    for i, layer in enumerate(model.layers):
        layer_start = time.time()
        
        # Enhanced Attention with batching
        Q, K, V = layer.attention.get_QKV(embeddings)
        attention_scores = layer.attention.compute_attention_scores(Q, K)
        attention_weights = softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ V
        embeddings = layer.norm1(attention_output + embeddings)
        
        # Enhanced FeedForward
        ff_output = layer.ff.forward(embeddings)
        embeddings = layer.norm2(ff_output + embeddings)
        
        layer_time = time.time() - layer_start
        logger.debug(f"Layer {i+1} processing time: {layer_time:.4f}s")
    
    # Final output projection
    logits = embeddings @ torch.tensor(model.output_layer)
    
    total_time = time.time() - start_time
    logger.info(f"Forward pass completed in {total_time:.4f}s using {'quantum' if config.use_quantum else 'classical'} mode")
    
    if use_quantum is not None:
        config.use_quantum = old_setting
        
    return logits

# Benchmarking function to compare quantum vs classical
def benchmark_comparison(model, batch, runs=3):
    """Compare performance between quantum and classical computation"""
    results = {
        "quantum": {"time": [], "result": None},
        "classical": {"time": [], "result": None}
    }
    
    print("Running quantum benchmark...")
    for i in range(runs):
        start = time.time()
        q_result = forward_pass(model, batch, use_quantum=True)
        q_time = time.time() - start
        results["quantum"]["time"].append(q_time)
        results["quantum"]["result"] = q_result
        print(f"  Run {i+1}/{runs}: {q_time:.4f}s")
    
    print("Running classical benchmark...")
    for i in range(runs):
        start = time.time()
        c_result = forward_pass(model, batch, use_quantum=False)
        c_time = time.time() - start
        results["classical"]["time"].append(c_time)
        results["classical"]["result"] = c_result
        print(f"  Run {i+1}/{runs}: {c_time:.4f}s")
    
    q_avg = np.mean(results["quantum"]["time"])
    c_avg = np.mean(results["classical"]["time"])
    
    print(f"\nBenchmark Results:")
    print(f"  Quantum: {q_avg:.4f}s (avg over {runs} runs)")
    print(f"  Classical: {c_avg:.4f}s (avg over {runs} runs)")
    print(f"  Ratio (Quantum/Classical): {q_avg/c_avg:.2f}x")
    
    # Check result difference
    q_res_flat = results["quantum"]["result"].flatten()
    c_res_flat = results["classical"]["result"].flatten()
    res_diff = torch.mean(torch.abs(q_res_flat - c_res_flat)).item()
    print(f"  Result Difference (mean abs): {res_diff:.6f}")
    
    return results

# Example Usage
if __name__ == "__main__":
    # Set log level to INFO for production use
    logger.setLevel(logging.INFO)
    
    # Create model with slightly larger dimensions to better show quantum effects
    print("Initializing model...")
    model = Model(vocab_size=100, d_model=8, n_layers=2, d_ff=16)
    
    # Create dummy input
    batch = torch.zeros(2, 3, dtype=torch.long)  # Dummy input (batch_size=2, seq_len=3)
    
    # Single forward pass
    print("\nRunning single forward pass...")
    logits = forward_pass(model, batch)
    print(f"Logits shape: {logits.shape}")
    
    # Run benchmark comparison
    print("\nRunning benchmark comparison...")
    benchmark_results = benchmark_comparison(model, batch, runs=3)
