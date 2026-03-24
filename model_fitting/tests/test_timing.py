"""
Timing test for tree_search_parallel scaling with different numbers of threads.

Tests the performance of MultiThreadedFitter with different n_threads values
to measure parallelization scaling.
"""
import sys
from pathlib import Path
import os
import glob
import random
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tree_search_parallel import MultiThreadedFitter, create_tree_search_config


def find_data_folder():
    """Find a data folder with split files."""
    possible_paths = [
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-03-13",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2024-02-14",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-02-20",
    ]
    
    for path in possible_paths:
        if glob.glob(f"{path}/split_*.csv"):
            return path
    
    # Try to find any folder with split files
    base_path = "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling"
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and glob.glob(f"{item_path}/split_*.csv"):
                return item_path
    
    return None


def test_timing(data_folder=None, n_samples=50, n_threads_list=None, manual_seed=1, verbose=True):
    """
    Test MultiThreadedFitter performance scaling with different thread counts.
    
    Args:
        data_folder: Path to directory containing split_*.csv files
        n_samples: Number of data samples to use for testing (default: 50)
        n_threads_list: List of thread counts to test (default: [1, 2, 4, 8])
        manual_seed: Random seed for reproducibility
        verbose: Print detailed output
    
    Returns:
        dict with timing results for each thread count
    """
    # Set seed for reproducibility
    random.seed(manual_seed)
    
    # Default thread counts to test
    if n_threads_list is None:
        n_threads_list = [1, 2, 4, 8]
    
    # Find data folder if not provided
    if data_folder is None:
        data_folder = find_data_folder()
    
    if data_folder is None:
        print("❌ ERROR: Could not find data folder with split_*.csv files")
        return None
    
    if verbose:
        print("=" * 80)
        print("TIMING TEST: MultiThreadedFitter scaling with n_threads")
        print("=" * 80)
        print(f"Data folder: {data_folder}")
        print(f"Number of samples: {n_samples}")
        print(f"Thread counts to test: {n_threads_list}")
        print(f"Manual seed: {manual_seed}")
        print()
    
    # Load data from split_0.csv
    try:
        split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
        if not split_files:
            print(f"❌ ERROR: Could not find split_*.csv files in {data_folder}")
            return None
        
        data_file = split_files[0]
        full_data = pd.read_csv(data_file)
        
        if len(full_data) < n_samples:
            print(f"⚠️  Warning: Requested {n_samples} samples but only {len(full_data)} available.")
            print(f"   Using all {len(full_data)} samples instead.")
            test_data = full_data.copy()
        else:
            test_data = full_data.sample(n=n_samples, random_state=manual_seed).copy()
        
        if verbose:
            print(f"Loaded {len(full_data)} samples from {os.path.basename(data_file)}")
            print(f"Using {len(test_data)} samples for testing")
            print()
    
    except Exception as e:
        print(f"❌ ERROR: Could not load data from {data_folder}")
        print(f"   Error: {e}")
        return None
    
    # Create config once (shared across all thread counts)
    config = create_tree_search_config()
    
    # BADS options - limit to ~10 function evaluations for faster testing
    def get_bads_options():
        return {
            'uncertainty_handling': False,
            'noise_final_samples': 0,
            'max_fun_evals': 10,   # Stop after ~10 function evaluations
            'max_iter': 20,        # Allow enough iterations
            'tol_mesh': 1e-6,      # Standard tolerance
            'tol_fun': 1e-4        # Standard tolerance
        }
    
    results = {}
    
    for n_threads in n_threads_list:
        if verbose:
            print("=" * 80)
            print(f"Testing MultiThreadedFitter with n_threads={n_threads}")
            print("=" * 80)
        
        # Setup fitter
        fitter = MultiThreadedFitter(config, n_repeats=1, verbose=False, n_threads=n_threads)
        
        # Measure time
        start_time = time.time()
        
        try:
            bads_options = get_bads_options()
            
            fitted_params, final_LL = fitter.fit(
                test_data.copy(),  # Use copy to ensure same data each time
                manual_seed=manual_seed,
                bads_options=bads_options
            )
            
            elapsed_time = time.time() - start_time
            
            results[n_threads] = {
                'time': elapsed_time,
                'success': True,
                'fitted_params': fitted_params,
                'final_LL': final_LL,
                'n_samples': len(test_data),
                'n_iterations': fitter.iteration_count
            }
            
            if verbose:
                print(f"\n✅ Completed in {elapsed_time:.2f} seconds")
                print(f"   BADS iterations: {fitter.iteration_count}")
                if final_LL is not None:
                    print(f"   Final log-likelihood: {final_LL.sum():.4f}")
                    print(f"   Average LL per sample: {final_LL.mean():.4f}")
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[n_threads] = {
                'time': elapsed_time,
                'success': False,
                'error': str(e),
                'n_samples': len(test_data)
            }
            
            if verbose:
                print(f"\n❌ Failed after {elapsed_time:.2f} seconds")
                print(f"   Error: {e}")
        
        if verbose:
            print()
    
    # Print summary
    print("=" * 80)
    print("TIMING TEST SUMMARY")
    print("=" * 80)
    
    successful_runs = [t for t in n_threads_list if results[t]['success']]
    
    if successful_runs:
        baseline_time = results[successful_runs[0]]['time'] if successful_runs else None
        
        print(f"\n{'n_threads':>12} | {'Time (s)':>12} | {'Speedup':>10} | {'Efficiency':>12}")
        print("-" * 60)
        
        for n_threads in successful_runs:
            r = results[n_threads]
            time_val = r['time']
            speedup = baseline_time / time_val if baseline_time else 1.0
            efficiency = speedup / n_threads if n_threads > 0 else 0.0
            print(f"{n_threads:>12} | {time_val:>12.2f} | {speedup:>10.2f}x | {efficiency:>12.2%}")
    
    failed_runs = [t for t in n_threads_list if not results[t]['success']]
    if failed_runs:
        print("\n❌ Failed runs:")
        for n_threads in failed_runs:
            r = results[n_threads]
            print(f"   n_threads={n_threads}: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MultiThreadedFitter scaling with different thread counts')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of samples to use for testing (default: 50)')
    parser.add_argument('--data-folder', type=str, default=None,
                        help='Path to directory containing split_*.csv files')
    parser.add_argument('--n-threads', type=int, nargs='+', default=None,
                        help='Thread counts to test (default: 1 2 4 8)')
    
    args = parser.parse_args()
    
    # Run the timing test
    results = test_timing(
        data_folder=args.data_folder,
        n_samples=args.n_samples,
        n_threads_list=args.n_threads,
        verbose=True
    )
    
    if results:
        print("\n✅ Timing test completed!")
    else:
        print("\n❌ Timing test failed to run.")
