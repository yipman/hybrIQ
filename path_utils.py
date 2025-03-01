"""
Path utilities for HybrIQ to ensure proper path handling across different operating systems.
"""

import os
import platform
import logging

logger = logging.getLogger('HybrIQ.path_utils')

def get_project_root():
    """Get the HybrIQ project root directory."""
    # This assumes this file is in the project root
    return os.path.dirname(os.path.abspath(__file__))

def normalize_path(path):
    """
    Normalize a path string to be compatible with the current operating system.
    
    Args:
        path (str): Path to normalize
        
    Returns:
        str: Normalized path
    """
    # Convert Unix style paths to Windows if needed
    if platform.system() == 'Windows' and path.startswith('/'):
        # Handle cases like '/d:/path' -> 'D:/path'
        if len(path) > 3 and path[1].isalpha() and path[2] == ':':
            path = path[1].upper() + path[2:]
        # For other cases, remove the leading slash
        else:
            path = path[1:] if path.startswith('/') else path
    
    # Use os.path.normpath to clean up the path
    return os.path.normpath(path)

def ensure_dir_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to check/create
        
    Returns:
        str: Normalized path to the directory
    """
    # Normalize the path first
    norm_dir = normalize_path(directory)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(norm_dir):
        try:
            os.makedirs(norm_dir, exist_ok=True)
            logger.info(f"Created directory: {norm_dir}")
        except Exception as e:
            logger.error(f"Failed to create directory {norm_dir}: {str(e)}")
    
    return norm_dir

def join_paths(*paths):
    """
    Join paths and normalize the result.
    
    Args:
        *paths: Path components to join
        
    Returns:
        str: Normalized joined path
    """
    # Join the paths and normalize
    return normalize_path(os.path.join(*paths))

def get_default_results_dir():
    """
    Get the default directory for storing results.
    
    Returns:
        str: Path to results directory
    """
    results_dir = join_paths(get_project_root(), 'results')
    return ensure_dir_exists(results_dir)

def get_benchmark_dir():
    """
    Get the directory for storing benchmark results.
    
    Returns:
        str: Path to benchmark results directory
    """
    benchmark_dir = join_paths(get_project_root(), 'benchmark_results')
    return ensure_dir_exists(benchmark_dir)

if __name__ == "__main__":
    # Test path utilities
    print(f"Project Root: {get_project_root()}")
    print(f"Normalized Path: {normalize_path('/d:/HybrIQ/benchmark_results')}")
    print(f"Default Results Dir: {get_default_results_dir()}")
    print(f"Benchmark Dir: {get_benchmark_dir()}")
