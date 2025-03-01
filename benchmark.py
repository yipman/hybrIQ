import torch
import numpy as np  # Fix: Use numpy instead of numpy 
import time
import matplotlib.pyplot as plt
from model import Model, forward_pass, config, HybridConfig
import logging
from tqdm import tqdm
from qiskit_aer import Aer  # Fixed import for Aer
import os
import json
from datetime import datetime
import pandas as pd
from quantum_utils import quantum_resource_estimator, compare_classical_quantum

# Setup logging - fix typo in format string
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HybrIQ.benchmark')

class BenchmarkSuite:
    def __init__(self, save_dir=None):
        # Fix: Use a Windows-compatible path format and default to current directory if not specified
        if save_dir is None:
            # Use current directory + benchmark_results
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
        
        self.save_dir = save_dir
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        
        logger.info(f"Benchmark results will be saved to {self.save_dir}")
        
    def run_size_scaling_test(self, sizes=[4, 8, 16, 32], n_layers=1, trials=3):
        """Test how performance scales with model dimension"""
        results = {
            'sizes': sizes,
            'quantum_times': [],
            'quantum_stds': [],
            'classical_times': [],
            'classical_stds': [],
            'speedup_ratios': []
        }
        
        for size in tqdm(sizes, desc="Testing model sizes"):
            quantum_times = []
            classical_times = []
            
            for _ in range(trials):
                # Create model with this dimension
                model = Model(vocab_size=100, d_model=size, n_layers=n_layers, d_ff=size*4)
                
                # Dummy input
                batch = torch.zeros(2, 3, dtype=torch.long)
                
                # Quantum timing
                start = time.time()
                forward_pass(model, batch, use_quantum=True)
                q_time = time.time() - start
                quantum_times.append(q_time)
                
                # Classical timing
                start = time.time()
                forward_pass(model, batch, use_quantum=False)
                c_time = time.time() - start
                classical_times.append(c_time)
                
            # Compute stats
            q_mean = np.mean(quantum_times)
            q_std = np.std(quantum_times)
            c_mean = np.mean(classical_times)
            c_std = np.std(classical_times)
            speedup = c_mean / q_mean if q_mean > 0 else 0
            
            # Store results
            results['quantum_times'].append(q_mean)
            results['quantum_stds'].append(q_std)
            results['classical_times'].append(c_mean)
            results['classical_stds'].append(c_std)
            results['speedup_ratios'].append(speedup)
            
            logger.info(f"Size {size}: Quantum={q_mean:.4f}s, Classical={c_mean:.4f}s, Speedup={speedup:.2f}x")
        
        self.results['size_scaling'] = results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"size_scaling_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Size scaling test results saved to {save_path}")
        return results
        
    def run_layer_scaling_test(self, n_layers_list=[1, 2, 4, 8], d_model=8, trials=3):
        """Test how performance scales with number of transformer layers"""
        results = {
            'n_layers': n_layers_list,
            'quantum_times': [],
            'quantum_stds': [],
            'classical_times': [],
            'classical_stds': [],
            'speedup_ratios': []
        }
        
        for n_layers in tqdm(n_layers_list, desc="Testing layer scaling"):
            quantum_times = []
            classical_times = []
            
            for _ in range(trials):
                # Create model with this number of layers
                model = Model(vocab_size=100, d_model=d_model, n_layers=n_layers, d_ff=d_model*4)
                
                # Dummy input
                batch = torch.zeros(2, 3, dtype=torch.long)
                
                # Quantum timing
                start = time.time()
                forward_pass(model, batch, use_quantum=True)
                q_time = time.time() - start
                quantum_times.append(q_time)
                
                # Classical timing
                start = time.time()
                forward_pass(model, batch, use_quantum=False)
                c_time = time.time() - start
                classical_times.append(c_time)
                
            # Compute stats
            q_mean = np.mean(quantum_times)
            q_std = np.std(quantum_times)
            c_mean = np.mean(classical_times)
            c_std = np.std(classical_times)
            speedup = c_mean / q_mean if q_mean > 0 else 0
            
            # Store results
            results['quantum_times'].append(q_mean)
            results['quantum_stds'].append(q_std)
            results['classical_times'].append(c_mean)
            results['classical_stds'].append(c_std)
            results['speedup_ratios'].append(speedup)
            
            logger.info(f"Layers {n_layers}: Quantum={q_mean:.4f}s, Classical={c_mean:.4f}s, Speedup={speedup:.2f}x")
        
        self.results['layer_scaling'] = results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"layer_scaling_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Layer scaling test results saved to {save_path}")
        return results
    
    def run_noise_sensitivity_test(self, noise_levels=[0.0, 0.01, 0.05, 0.1], d_model=8, n_layers=2, trials=3):
        """Test how quantum performance degrades with increasing noise levels"""
        # Save original noise model
        original_noise = config.noise_model
        
        results = {
            'noise_levels': noise_levels,
            'quantum_times': [],
            'quantum_stds': [],
            'quantum_error_rates': []
        }
        
        # First get baseline classical results for comparison
        model = Model(vocab_size=100, d_model=d_model, n_layers=n_layers, d_ff=d_model*4)
        batch = torch.zeros(2, 3, dtype=torch.long)
        classical_outputs = forward_pass(model, batch, use_quantum=False)
        
        for noise in tqdm(noise_levels, desc="Testing noise sensitivity"):
            quantum_times = []
            error_rates = []
            
            # Create simple depolarizing noise model with this error rate
            if noise > 0:
                # Fix: Update import to use qiskit_aer.noise instead of qiskit.providers.aer.noise
                from qiskit_aer.noise import NoiseModel
                from qiskit_aer.noise import depolarizing_error
                
                noise_model = NoiseModel()
                error = depolarizing_error(noise, 1)
                noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
                config.noise_model = noise_model
                config.initialize_backend()
            else:
                config.noise_model = None
                config.initialize_backend()
            
            for _ in range(trials):
                # Re-initialize model to ensure consistency
                model = Model(vocab_size=100, d_model=d_model, n_layers=n_layers, d_ff=d_model*4)
                
                # Quantum timing and output
                start = time.time()
                quantum_outputs = forward_pass(model, batch, use_quantum=True)
                q_time = time.time() - start
                quantum_times.append(q_time)
                
                # Calculate error compared to classical
                error_rate = torch.mean(torch.abs(quantum_outputs - classical_outputs)).item()
                error_rates.append(error_rate)
            
            # Compute stats
            q_mean = np.mean(quantum_times)
            q_std = np.std(quantum_times)
            err_mean = np.mean(error_rates)
            
            # Store results
            results['quantum_times'].append(q_mean)
            results['quantum_stds'].append(q_std)
            results['quantum_error_rates'].append(err_mean)
            
            logger.info(f"Noise {noise}: Quantum={q_mean:.4f}s, Error Rate={err_mean:.6f}")
        
        # Restore original noise model
        config.noise_model = original_noise
        config.initialize_backend()
        
        self.results['noise_sensitivity'] = results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"noise_sensitivity_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in results.items()}, f, indent=2)
        
        logger.info(f"Noise sensitivity test results saved to {save_path}")
        return results
    
    def run_shot_count_test(self, shots_list=[128, 512, 1024, 4096], d_model=8, n_layers=1, trials=3):
        """Test how shot count affects performance and accuracy"""
        # Save original shot count
        original_shots = config.shots
        
        results = {
            'shot_counts': shots_list,
            'quantum_times': [],
            'quantum_stds': [],
            'quantum_error_rates': []
        }
        
        # First get baseline classical results for comparison
        model = Model(vocab_size=100, d_model=d_model, n_layers=n_layers, d_ff=d_model*4)
        batch = torch.zeros(2, 3, dtype=torch.long)
        classical_outputs = forward_pass(model, batch, use_quantum=False)
        
        for shots in tqdm(shots_list, desc="Testing shot counts"):
            quantum_times = []
            error_rates = []
            
            # Update shot count
            config.shots = shots
            config.initialize_backend()
            
            for _ in range(trials):
                # Re-initialize model to ensure consistency
                model = Model(vocab_size=100, d_model=d_model, n_layers=n_layers, d_ff=d_model*4)
                
                # Quantum timing and output
                start = time.time()
                quantum_outputs = forward_pass(model, batch, use_quantum=True)
                q_time = time.time() - start
                quantum_times.append(q_time)
                
                # Calculate error compared to classical
                error_rate = torch.mean(torch.abs(quantum_outputs - classical_outputs)).item()
                error_rates.append(error_rate)
            
            # Compute stats
            q_mean = np.mean(quantum_times)
            q_std = np.std(quantum_times)
            err_mean = np.mean(error_rates)
            
            # Store results
            results['quantum_times'].append(q_mean)
            results['quantum_stds'].append(q_std)
            results['quantum_error_rates'].append(err_mean)
            
            logger.info(f"Shots {shots}: Quantum={q_mean:.4f}s, Error Rate={err_mean:.6f}")
        
        # Restore original shot count
        config.shots = original_shots
        config.initialize_backend()
        
        self.results['shot_count'] = results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"shot_count_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in results.items()}, f, indent=2)
        
        logger.info(f"Shot count test results saved to {save_path}")
        return results
    
    def plot_size_scaling(self, results=None):
        """Plot how performance scales with model size"""
        if results is None:
            if 'size_scaling' not in self.results:
                logger.error("No size scaling results found. Run run_size_scaling_test first.")
                return
            results = self.results['size_scaling']
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        ax1.errorbar(results['sizes'], results['quantum_times'], yerr=results['quantum_stds'], 
                    label='Quantum', marker='o', linestyle='-', capsize=5)
        ax1.errorbar(results['sizes'], results['classical_times'], yerr=results['classical_stds'], 
                    label='Classical', marker='s', linestyle='-', capsize=5)
        ax1.set_xlabel('Model Dimension (d_model)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Performance vs Model Size')
        ax1.legend()
        ax1.grid(True)
        
        # Speedup ratio
        ax2.plot(results['sizes'], results['speedup_ratios'], marker='o', linestyle='-')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Model Dimension (d_model)')
        ax2.set_ylabel('Speedup Ratio (Classical/Quantum)')
        ax2.set_title('Quantum Speedup vs Model Size')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"size_scaling_plot_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Size scaling plot saved to {save_path}")
        
        return fig
        
    def plot_layer_scaling(self, results=None):
        """Plot how performance scales with number of layers"""
        if results is None:
            if 'layer_scaling' not in self.results:
                logger.error("No layer scaling results found. Run run_layer_scaling_test first.")
                return
            results = self.results['layer_scaling']
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        ax1.errorbar(results['n_layers'], results['quantum_times'], yerr=results['quantum_stds'], 
                    label='Quantum', marker='o', linestyle='-', capsize=5)
        ax1.errorbar(results['n_layers'], results['classical_times'], yerr=results['classical_stds'], 
                    label='Classical', marker='s', linestyle='-', capsize=5)
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Performance vs Number of Layers')
        ax1.legend()
        ax1.grid(True)
        
        # Speedup ratio
        ax2.plot(results['n_layers'], results['speedup_ratios'], marker='o', linestyle='-')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('Speedup Ratio (Classical/Quantum)')
        ax2.set_title('Quantum Speedup vs Number of Layers')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"layer_scaling_plot_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Layer scaling plot saved to {save_path}")
        
        return fig
        
    def plot_noise_sensitivity(self, results=None):
        """Plot how noise affects quantum performance"""
        if results is None:
            if 'noise_sensitivity' not in self.results:
                logger.error("No noise sensitivity results found. Run run_noise_sensitivity_test first.")
                return
            results = self.results['noise_sensitivity']
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance degradation
        ax1.errorbar(results['noise_levels'], results['quantum_times'], yerr=results['quantum_stds'],
                    marker='o', linestyle='-', capsize=5)
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Quantum Execution Time vs Noise Level')
        ax1.grid(True)
        
        # Error rate
        ax2.plot(results['noise_levels'], results['quantum_error_rates'], marker='o', linestyle='-')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Error Rate vs Classical')
        ax2.set_title('Quantum Accuracy vs Noise Level')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"noise_sensitivity_plot_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Noise sensitivity plot saved to {save_path}")
        
        return fig
        
    def plot_shot_count(self, results=None):
        """Plot how shot count affects quantum performance"""
        if results is None:
            if 'shot_count' not in self.results:
                logger.error("No shot count results found. Run run_shot_count_test first.")
                return
            results = self.results['shot_count']
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs shots
        ax1.errorbar(results['shot_counts'], results['quantum_times'], yerr=results['quantum_stds'],
                    marker='o', linestyle='-', capsize=5)
        ax1.set_xlabel('Shot Count')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Quantum Execution Time vs Shot Count')
        ax1.set_xscale('log')
        ax1.grid(True)
        
        # Error rate vs shots
        ax2.plot(results['shot_counts'], results['quantum_error_rates'], marker='o', linestyle='-')
        ax2.set_xlabel('Shot Count')
        ax2.set_ylabel('Error Rate vs Classical')
        ax2.set_title('Quantum Accuracy vs Shot Count')
        ax2.set_xscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"shot_count_plot_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Shot count plot saved to {save_path}")
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive benchmark report as a DataFrame"""
        report_data = {
            'Test': [],
            'Parameter': [],
            'Value': [],
            'Quantum Time (s)': [],
            'Classical Time (s)': [],
            'Speedup': [],
            'Error Rate': []
        }
        
        # Size scaling data
        if 'size_scaling' in self.results:
            results = self.results['size_scaling']
            for i, size in enumerate(results['sizes']):
                report_data['Test'].append('Size Scaling')
                report_data['Parameter'].append('Model Dimension')
                report_data['Value'].append(size)
                report_data['Quantum Time (s)'].append(f"{results['quantum_times'][i]:.4f} ± {results['quantum_stds'][i]:.4f}")
                report_data['Classical Time (s)'].append(f"{results['classical_times'][i]:.4f} ± {results['classical_stds'][i]:.4f}")
                report_data['Speedup'].append(f"{results['speedup_ratios'][i]:.2f}x")
                report_data['Error Rate'].append('N/A')
        
        # Layer scaling data
        if 'layer_scaling' in self.results:
            results = self.results['layer_scaling']
            for i, layers in enumerate(results['n_layers']):
                report_data['Test'].append('Layer Scaling')
                report_data['Parameter'].append('Number of Layers')
                report_data['Value'].append(layers)
                report_data['Quantum Time (s)'].append(f"{results['quantum_times'][i]:.4f} ± {results['quantum_stds'][i]:.4f}")
                report_data['Classical Time (s)'].append(f"{results['classical_times'][i]:.4f} ± {results['classical_stds'][i]:.4f}")
                report_data['Speedup'].append(f"{results['speedup_ratios'][i]:.2f}x")
                report_data['Error Rate'].append('N/A')
        
        # Noise sensitivity data
        if 'noise_sensitivity' in self.results:
            results = self.results['noise_sensitivity']
            for i, noise in enumerate(results['noise_levels']):
                report_data['Test'].append('Noise Sensitivity')
                report_data['Parameter'].append('Noise Level')
                report_data['Value'].append(noise)
                report_data['Quantum Time (s)'].append(f"{results['quantum_times'][i]:.4f} ± {results['quantum_stds'][i]:.4f}")
                report_data['Classical Time (s)'].append('N/A')
                report_data['Speedup'].append('N/A')
                report_data['Error Rate'].append(f"{results['quantum_error_rates'][i]:.6f}")
        
        # Shot count data
        if 'shot_count' in self.results:
            results = self.results['shot_count']
            for i, shots in enumerate(results['shot_counts']):
                report_data['Test'].append('Shot Count')
                report_data['Parameter'].append('Number of Shots')
                report_data['Value'].append(shots)
                report_data['Quantum Time (s)'].append(f"{results['quantum_times'][i]:.4f} ± {results['quantum_stds'][i]:.4f}")
                report_data['Classical Time (s)'].append('N/A')
                report_data['Speedup'].append('N/A')
                report_data['Error Rate'].append(f"{results['quantum_error_rates'][i]:.6f}")
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"benchmark_report_{timestamp}.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"Benchmark report saved to {save_path}")
        
        return df

def run_standard_benchmark_suite(small=True):
    """Run a standard set of benchmarks"""
    benchmark = BenchmarkSuite()
    
    if small:
        # Small benchmarks for quick testing
        logger.info("Running small benchmark suite for quick testing")
        benchmark.run_size_scaling_test(sizes=[4, 8], n_layers=1, trials=2)
        benchmark.run_layer_scaling_test(n_layers_list=[1, 2], d_model=4, trials=2)
        benchmark.run_shot_count_test(shots_list=[128, 512], d_model=4, trials=2)
    else:
        # Full benchmark suite
        logger.info("Running full benchmark suite (this may take a while)")
        benchmark.run_size_scaling_test(sizes=[4, 8, 16, 32], n_layers=1, trials=3)
        benchmark.run_layer_scaling_test(n_layers_list=[1, 2, 4], d_model=8, trials=3)
        benchmark.run_noise_sensitivity_test(noise_levels=[0.0, 0.01, 0.05, 0.1], d_model=8, trials=3)
        benchmark.run_shot_count_test(shots_list=[128, 512, 1024, 4096], d_model=8, trials=3)
    
    # Generate plots
    benchmark.plot_size_scaling()
    benchmark.plot_layer_scaling()
    if not small:
        benchmark.plot_noise_sensitivity()
        benchmark.plot_shot_count()
    
    # Generate report
    report = benchmark.generate_report()
    print("\nBenchmark Report Summary:")
    print(report)
    
    return benchmark

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HybrIQ Benchmarking Suite')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite (slower but more comprehensive)')
    parser.add_argument('--size', action='store_true', help='Run only size scaling benchmark')
    parser.add_argument('--layers', action='store_true', help='Run only layer scaling benchmark')
    parser.add_argument('--shots', action='store_true', help='Run only shot count benchmark')
    parser.add_argument('--noise', action='store_true', help='Run only noise sensitivity benchmark')
    
    args = parser.parse_args()
    
    benchmark = BenchmarkSuite()
    
    if args.size:
        benchmark.run_size_scaling_test()
        benchmark.plot_size_scaling()
    
    if args.layers:
        benchmark.run_layer_scaling_test()
        benchmark.plot_layer_scaling()
    
    if args.shots:
        benchmark.run_shot_count_test()
        benchmark.plot_shot_count()
    
    if args.noise:
        benchmark.run_noise_sensitivity_test()
        benchmark.plot_noise_sensitivity()
    
    # If no specific tests requested, run standard suite
    if not (args.size or args.layers or args.shots or args.noise):
        benchmark = run_standard_benchmark_suite(small=not args.full)
    
    # Always generate a report
    report = benchmark.generate_report()
    print("\nBenchmark Report Summary:")
    print(report.head(10))
    if len(report) > 10:
        print(f"... and {len(report) - 10} more rows")

"""
Benchmark script for HybrIQ quantum-classical hybrid model.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from model import Model, forward_pass, config, HybridConfig
from circuit_optimizer import optimize_circuit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HybrIQ.benchmark')

def run_benchmark(model_dims=[4, 8, 16], batch_size=2, seq_len=3, runs=3):
    """
    Run benchmarks comparing quantum vs classical execution with different model dimensions.
    
    Args:
        model_dims: List of model dimensions to test
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        runs: Number of runs for each configuration
    
    Returns:
        Dictionary of benchmark results
    """
    results = {
        "dimensions": model_dims,
        "quantum": {"time": [], "std": []},
        "classical": {"time": [], "std": []},
        "ratio": []
    }
    
    for dim in model_dims:
        logger.info(f"Benchmarking with dimension {dim}")
        
        # Create model with current dimension
        model = Model(vocab_size=100, d_model=dim, n_layers=2, d_ff=dim*4)
        
        # Create input
        batch = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Run quantum benchmark
        q_times = []
        for i in range(runs):
            start = time.time()
            forward_pass(model, batch, use_quantum=True)
            q_times.append(time.time() - start)
            logger.info(f"  Quantum run {i+1}/{runs}: {q_times[-1]:.4f}s")
        
        # Run classical benchmark
        c_times = []
        for i in range(runs):
            start = time.time()
            forward_pass(model, batch, use_quantum=False)
            c_times.append(time.time() - start)
            logger.info(f"  Classical run {i+1}/{runs}: {c_times[-1]:.4f}s")
        
        # Calculate statistics
        q_avg = np.mean(q_times)
        q_std = np.std(q_times)
        c_avg = np.mean(c_times)
        c_std = np.std(c_times)
        ratio = q_avg / c_avg
        
        # Store results
        results["quantum"]["time"].append(q_avg)
        results["quantum"]["std"].append(q_std)
        results["classical"]["time"].append(c_avg)
        results["classical"]["std"].append(c_std)
        results["ratio"].append(ratio)
        
        logger.info(f"  Dimension {dim} results:")
        logger.info(f"    Quantum: {q_avg:.4f}s ± {q_std:.4f}")
        logger.info(f"    Classical: {c_avg:.4f}s ± {c_std:.4f}")
        logger.info(f"    Ratio (Q/C): {ratio:.2f}x")
    
    return results

def compare_optimization_levels(levels=[1, 2, 3], model_dim=8, batch_size=2, seq_len=3, runs=3):
    """
    Compare different optimization levels for quantum circuits.
    
    Args:
        levels: List of optimization levels to test
        model_dim: Model dimension for testing
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        runs: Number of runs for each configuration
    
    Returns:
        Dictionary of benchmark results
    """
    results = {
        "levels": levels,
        "times": [],
        "std": [],
    }
    
    original_level = config.circuit_optimization_level
    
    for level in levels:
        logger.info(f"Testing optimization level {level}")
        
        # Set optimization level
        config.circuit_optimization_level = level
        
        # Create model
        model = Model(vocab_size=100, d_model=model_dim, n_layers=2, d_ff=model_dim*4)
        
        # Create input
        batch = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Clear the lru_cache to ensure circuits are rebuilt
        from model import create_inner_product_circuit, create_matrix_vector_circuit
        create_inner_product_circuit.cache_clear()
        create_matrix_vector_circuit.cache_clear()
        
        # Run benchmark
        times = []
        for i in range(runs):
            start = time.time()
            forward_pass(model, batch, use_quantum=True)
            times.append(time.time() - start)
            logger.info(f"  Run {i+1}/{runs}: {times[-1]:.4f}s")
        
        # Calculate statistics
        avg = np.mean(times)
        std = np.std(times)
        
        # Store results
        results["times"].append(avg)
        results["std"].append(std)
        
        logger.info(f"  Level {level} results: {avg:.4f}s ± {std:.4f}")
    
    # Restore original level
    config.circuit_optimization_level = original_level
    
    return results

def plot_benchmark_results(results):
    """
    Plot benchmark results comparing quantum vs classical execution.
    
    Args:
        results: Results dictionary from run_benchmark function
    """
    plt.figure(figsize=(10, 6))
    
    x = results["dimensions"]
    q_y = results["quantum"]["time"]
    q_err = results["quantum"]["std"]
    c_y = results["classical"]["time"]
    c_err = results["classical"]["std"]
    
    plt.errorbar(x, q_y, yerr=q_err, marker='o', label='Quantum')
    plt.errorbar(x, c_y, yerr=c_err, marker='s', label='Classical')
    
    plt.xlabel('Model Dimension')
    plt.ylabel('Execution Time (s)')
    plt.title('Quantum vs Classical Execution Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add ratio as text
    for i, dim in enumerate(x):
        plt.text(dim, max(q_y[i], c_y[i]) + 0.1, 
                f"Ratio: {results['ratio'][i]:.2f}x", 
                ha='center')
    
    plt.tight_layout()
    plt.savefig('quantum_classical_comparison.png')
    plt.close()
    
    logger.info("Saved comparison plot to 'quantum_classical_comparison.png'")

def plot_optimization_comparison(results):
    """
    Plot benchmark results for different optimization levels.
    
    Args:
        results: Results dictionary from compare_optimization_levels function
    """
    plt.figure(figsize=(8, 5))
    
    x = results["levels"]
    y = results["times"]
    err = results["std"]
    
    plt.errorbar(x, y, yerr=err, marker='o')
    
    plt.xlabel('Optimization Level')
    plt.ylabel('Execution Time (s)')
    plt.title('Circuit Optimization Level Performance')
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    
    plt.tight_layout()
    plt.savefig('optimization_levels_comparison.png')
    plt.close()
    
    logger.info("Saved optimization comparison plot to 'optimization_levels_comparison.png'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HybrIQ benchmarks')
    parser.add_argument('--benchmark-type', choices=['model-size', 'optimization'], 
                       default='model-size', help='Type of benchmark to run')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[4, 8, 16],
                       help='Model dimensions to test')
    parser.add_argument('--opt-levels', type=int, nargs='+', default=[1, 2, 3],
                       help='Optimization levels to test')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for each configuration')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for testing')
    parser.add_argument('--seq-len', type=int, default=3,
                       help='Sequence length for testing')
    
    args = parser.parse_args()
    
    if args.benchmark_type == 'model-size':
        logger.info("Running model size benchmark...")
        results = run_benchmark(
            model_dims=args.dimensions,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runs=args.runs
        )
        plot_benchmark_results(results)
    else:  # optimization benchmark
        logger.info("Running optimization level benchmark...")
        results = compare_optimization_levels(
            levels=args.opt_levels,
            model_dim=8,  # Fixed dimension for optimization comparison
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runs=args.runs
        )
        plot_optimization_comparison(results)
    
    logger.info("Benchmarking completed!")
