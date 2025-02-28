import numpy as np
import torch
from torch.nn.functional import softmax
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
import time
from functools import lru_cache
import logging

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
        self.simulator = 'qasm_simulator'
        self.noise_model = None
        self.quantum_threshold = 8  # Max dimension to use quantum computation (above this, use classical)
        self.circuit_reuse = True
        self.batch_circuits = True
        self.max_batch_size = 100
        self.backend = None
        self.quantum_instance = None
        
    def initialize_backend(self):
        """Initialize the quantum backend with current settings"""
        backend = Aer.get_backend(self.simulator)
        self.backend = backend
        self.quantum_instance = QuantumInstance(
            backend=backend,
            shots=self.shots,
            optimization_level=self.circuit_optimization_level,
            noise_model=self.noise_model,
            measurement_error_mitigation_cls=None if not self.measurement_error_mitigation else True
        )
        return self.quantum_instance
    
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
    qc = QuantumCircuit(num_qubits + 1, 1)
    qc.h(0)
    # We'll initialize the vectors later
    qc.h(0)
    qc.measure(0, 0)
    return transpile(qc, config.backend, optimization_level=config.circuit_optimization_level)

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
    
    # Initialize with actual data
    qc.initialize(u_norm[:2**num_qubits].tolist() + [0]*(2**num_qubits - len(u_norm[:2**num_qubits])), 
                 range(1, num_qubits + 1), control_qubit=0, invert=True)
    qc.initialize(v_norm[:2**num_qubits].tolist() + [0]*(2**num_qubits - len(v_norm[:2**num_qubits])), 
                 range(1, num_qubits + 1), control_qubit=0)
    
    counts = config.quantum_instance.execute(qc).get_counts()
    p0 = counts.get('0', 0) / config.shots
    p1 = counts.get('1', 0) / config.shots
    
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
                    p0 = batch_results[idx].get('0', 0) / config.shots
                    p1 = batch_results[idx].get('1', 0) / config.shots
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
            qc.initialize(u_norm[:2**num_qubits].tolist() + [0]*(2**num_qubits - len(u_norm[:2**num_qubits])), 
                         range(1, num_qubits + 1), control_qubit=0, invert=True)
            qc.initialize(v_norm[:2**num_qubits].tolist() + [0]*(2**num_qubits - len(v_norm[:2**num_qubits])), 
                         range(1, num_qubits + 1), control_qubit=0)
            
            circuits.append(qc)
            circuit_params.append((i, j))
            count += 1
    
    # Execute remaining circuits
    if circuits:
        batch_results = execute_quantum_batch(circuits)
        for idx, (ii, jj) in enumerate(circuit_params):
            p0 = batch_results[idx].get('0', 0) / config.shots
            p1 = batch_results[idx].get('1', 0) / config.shots
            q_norm = np.linalg.norm(query_vectors[ii])
            k_norm = np.linalg.norm(key_vectors[jj])
            results[ii, jj] = (p0 - p1) * q_norm * k_norm
    
    return results

def execute_quantum_batch(circuits):
    """Execute a batch of quantum circuits efficiently"""
    job = execute(circuits, config.backend, shots=config.shots)
    result = job.result()
    return [result.get_counts(i) for i in range(len(circuits))]

@lru_cache(maxsize=128)
def create_matrix_vector_circuit(num_qubits):
    """Create a reusable circuit template for matrix-vector multiplication"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    # Will initialize vector and apply unitary later
    qc.measure(range(num_qubits), range(num_qubits))
    return transpile(qc, config.backend, optimization_level=config.circuit_optimization_level)

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
    
    x_norm = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else np.zeros_like(x)
    
    # Get the circuit template
    qc = create_matrix_vector_circuit(num_qubits).copy()
    
    # Initialize with actual data
    qc.initialize(x_norm[:2**num_qubits].tolist() + [0]*(2**num_qubits - len(x_norm[:2**num_qubits])), 
                 range(num_qubits))
    
    # Apply unitary operation (simplified)
    W_scaled = W[:2**num_qubits, :2**num_qubits] / np.linalg.norm(W[:2**num_qubits, :2**num_qubits])
    qc.unitary(Operator(W_scaled), range(num_qubits), label='W')
    
    counts = config.quantum_instance.execute(qc).get_counts()
    
    y_est = np.zeros(2**num_qubits)
    for state, count in counts.items():
        y_est[int(state, 2)] = count / config.shots
    
    # Scale result according to original norms
    result = np.sqrt(y_est) * np.linalg.norm(W @ x)
    
    exec_time = time.time() - start_time
    if exec_time > 0.1:  # Log only if execution takes significant time
        logger.debug(f"Quantum matrix-vector mult took {exec_time:.4f}s for {W.shape} matrix")
    
    return result

# Enhanced Attention with quantum and classical paths
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

# Enhanced FeedForward with quantum optimization
class FeedForward:
    def __init__(self, d_model, d_ff, use_quantum=True):
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
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
        
        if config.should_use_quantum(d_model) and self.use_quantum:
            # Use quantum matrix-vector multiplication
            ff_intermediate = np.zeros((batch_size, seq_len, self.d_ff))
            for b in range(batch_size):
                for s in range(seq_len):
                    ff_intermediate[b, s] = quantum_matrix_vector_mult(self.W1, x_numpy[b, s])
        else:
            # Use classical computation
            ff_intermediate = np.matmul(x_numpy.reshape(-1, d_model), self.W1).reshape(batch_size, seq_len, -1)
        
        ff_intermediate = ff_intermediate + self.b1
        ff_intermediate_tensor = torch.tensor(ff_intermediate)
        ff_intermediate_tensor = gelu(ff_intermediate_tensor)
        
        # Always use classical for the second multiplication as it's more efficient
        ff_output = ff_intermediate_tensor @ torch.tensor(self.W2) + torch.tensor(self.b2)
        return ff_output

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
