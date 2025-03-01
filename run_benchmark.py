"""
Simple script to run HybrIQ benchmarks with proper path handling.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HybrIQ.run')

def main():
    try:
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current directory: {current_dir}")
        
        # Create results directory with proper Windows path
        results_dir = os.path.join(current_dir, 'benchmark_results')
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
        
        # Import and run benchmark
        from benchmark import BenchmarkSuite, run_standard_benchmark_suite
        
        # Run the standard benchmark suite with the proper path
        benchmark = BenchmarkSuite(save_dir=results_dir)
        
        # Run a small benchmark by default
        run_small = True
        if len(sys.argv) > 1 and sys.argv[1].lower() == 'full':
            run_small = False
            
        results = run_standard_benchmark_suite(small=run_small)
        
        logger.info(f"Benchmark completed successfully. Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
