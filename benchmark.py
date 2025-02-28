import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from model import Model, forward_pass, config, HybridConfig
import logging
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
from quantum_utils import quantum_resource_estimator, compare_classical_quantum

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname=s - %(message)s')
logger = logging.getLogger('HybrIQ.benchmark')

class BenchmarkSuite:
    def __init__(self, save_dir='/d:/HybrIQ/benchmark_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        
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
                from qiskit.providers.aer.noise import NoiseModel
                from qiskit.providers.aer.noise import depolarizing_error
                
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
