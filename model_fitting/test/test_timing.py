"""
Thread timing test for tree_search fit performance.

Tests whether the threads parameter actually leads to performance gains.
Tests the same amount of data across different thread counts (1, 2, 4, 8, 16)
to see if increasing threads improves performance.
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import random
import time
import glob
import os
from tree_search import TreeSearch, Fitter, initialize_thread_pool


class FastFitter(Fitter):
    """Fitter subclass that uses fewer evaluation iterations for faster testing."""
    def evaluate(self, params, data: pd.DataFrame, n_iters=3):
        """Evaluates the log-likelihood with fewer iterations for faster testing."""
        from tqdm import tqdm
        import numpy as np
        print(f"Running evaluation with {n_iters} iterations...")
        return np.array([self.log_likelihood(params, data) for _ in tqdm(range(n_iters))], dtype=np.float32).mean(axis=0)


def test_timing(models_base_path=None, n_samples=10, manual_seed=1, verbose=True):
    """
    Test whether the threads parameter actually leads to performance gains.
    
    Tests the same amount of data (n_samples) across different thread counts
    to see if increasing threads improves performance.
    
    Args:
        models_base_path: Path to directory containing week* model folders
        n_samples: Number of data samples to use for each test (default: 10)
        manual_seed: Random seed for reproducibility
        verbose: Print detailed output
    
    Returns:
        dict with timing results for each thread count
    """
    # Set seed for reproducibility
    random.seed(manual_seed)
    
    # Default models base path
    if models_base_path is None:
        models_base_path = "../../monkey_4iar/analysis/data/processed/harry/models"
    
    # Find all week* directories
    week_pattern = os.path.join(models_base_path, "*week*")
    week_dirs = glob.glob(week_pattern)
    
    if not week_dirs:
        print(f"❌ ERROR: Could not find any week* directories in {models_base_path}")
        return None
    
    # Pick one at random
    selected_week = random.choice(week_dirs)
    
    if verbose:
        print("=" * 80)
        print("THREAD TIMING TEST: Performance across different thread counts")
        print("=" * 80)
        print(f"Models base path: {models_base_path}")
        print(f"Selected week directory: {selected_week}")
        print(f"Number of samples per test: {n_samples}")
        print(f"Manual seed: {manual_seed}")
        print()
    
    # Load data from just split_0.csv (smaller subset for faster testing)
    try:
        data_file = os.path.join(selected_week, "split_0.csv")
        if not os.path.exists(data_file):
            print(f"❌ ERROR: Could not find {data_file}")
            return None
        
        # Load only the first n_samples rows to speed up loading
        full_data = pd.read_csv(data_file, nrows=n_samples * 2)  # Load a bit extra in case we need it
        
        if len(full_data) < n_samples:
            print(f"⚠️  Warning: Requested {n_samples} samples but only {len(full_data)} available.")
            print(f"   Using all {len(full_data)} samples instead.")
            test_data = full_data.copy()
        else:
            test_data = full_data.sample(n=n_samples, random_state=manual_seed).copy()
        
        if verbose:
            print(f"Loaded {len(full_data)} samples from split_0.csv")
            print(f"Using {len(test_data)} samples for testing")
            print()
    
    except Exception as e:
        print(f"❌ ERROR: Could not load data from {selected_week}")
        print(f"   Error: {e}")
        return None
    
    # Thread counts to test
    thread_counts = [1, 2, 3, 6]
    
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
    
    for threads in thread_counts:
        if verbose:
            print("=" * 80)
            print(f"Testing with {threads} thread(s)")
            print("=" * 80)
        
        # Setup model and fitter
        model = TreeSearch()
        fitter = FastFitter(model, threads=threads, verbose=verbose)
        
        # Measure time
        start_time = time.time()
        
        try:
            # Only use manual_seed with single thread (required for reproducibility)
            fit_seed = manual_seed if threads == 1 else None
            bads_options = get_bads_options()
            
            fitted_params, final_LL = fitter.fit(
                test_data.copy(),  # Use copy to ensure same data each time
                manual_seed=fit_seed,
                bads_options=bads_options
            )
            
            elapsed_time = time.time() - start_time
            
            results[threads] = {
                'time': elapsed_time,
                'success': True,
                'fitted_params': fitted_params,
                'final_LL': final_LL,
                'n_samples': len(test_data)
            }
            
            if verbose:
                print(f"\n✅ Completed in {elapsed_time:.2f} seconds")
                print(f"   BADS iterations: {fitter.iteration_count}")
                if final_LL is not None:
                    print(f"   Final log-likelihood: {final_LL.sum():.4f}")
                    print(f"   Average LL per sample: {final_LL.mean():.4f}")
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[threads] = {
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
    print("THREAD TIMING TEST SUMMARY")
    print("=" * 80)
    
    successful_runs = [t for t in thread_counts if results[t]['success']]
    
    if successful_runs:
        print(f"\n✅ Successful runs (using {n_samples} samples each):")
        print(f"{'Threads':>10} {'Time (s)':>12} {'Speedup':>12} {'Efficiency':>12}")
        print("-" * 50)
        
        baseline_time = results[successful_runs[0]]['time'] if successful_runs else None
        
        for threads in successful_runs:
            r = results[threads]
            time_val = r['time']
            
            if baseline_time and threads > successful_runs[0]:
                speedup = baseline_time / time_val
                efficiency = speedup / threads * 100  # Efficiency as percentage
                print(f"{threads:>10} {time_val:>12.2f} {speedup:>12.2f}x {efficiency:>11.1f}%")
            else:
                print(f"{threads:>10} {time_val:>12.2f} {'baseline':>12} {'N/A':>12}")
        
        # Analysis
        if len(successful_runs) >= 2:
            print("\n📊 Performance Analysis:")
            for i in range(1, len(successful_runs)):
                t1, time1 = successful_runs[i-1], results[successful_runs[i-1]]['time']
                t2, time2 = successful_runs[i], results[successful_runs[i]]['time']
                speedup = time1 / time2 if time2 > 0 else float('inf')
                expected_speedup = t2 / t1
                efficiency = speedup / expected_speedup * 100 if expected_speedup > 0 else 0
                print(f"   {t1} → {t2} threads: {speedup:.2f}x speedup (expected {expected_speedup:.2f}x, efficiency: {efficiency:.1f}%)")
    
    failed_runs = [t for t in thread_counts if not results[t]['success']]
    if failed_runs:
        print("\n❌ Failed runs:")
        for threads in failed_runs:
            r = results[threads]
            print(f"   {threads:2d} threads: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test thread timing performance')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to use for testing (default: 10)')
    parser.add_argument('--models-path', type=str, default=None,
                        help='Path to directory containing week* model folders')
    
    args = parser.parse_args()
    
    # Run the thread timing test
    results = test_timing(n_samples=args.n_samples, models_base_path=args.models_path, verbose=True)
    
    if results:
        print("\n✅ Thread timing test completed!")
    else:
        print("\n❌ Thread timing test failed to run.")

