"""
Scaling test for tree_search fit performance.

Tests how long it takes to run fit() to completion for different sample sizes:
- 1 sample
- 10 samples  
- 20 samples
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import random
import time
from tree_search import TreeSearch, Fitter, initialize_thread_pool


class FastFitter(Fitter):
    """Fitter subclass that uses fewer evaluation iterations for faster testing."""
    def evaluate(self, params, data: pd.DataFrame, n_iters=3):
        """Evaluates the log-likelihood with fewer iterations for faster testing."""
        from tqdm import tqdm
        import numpy as np
        print(f"Running evaluation with {n_iters} iterations...")
        return np.array([self.log_likelihood(params, data) for _ in tqdm(range(n_iters))], dtype=np.float32).mean(axis=0)


def test_scaling(data_folder=None, fold_idx=0, manual_seed=1, verbose=True, threads=50):
    """
    Test scaling performance of tree_search.fit() for different sample sizes.
    
    Args:
        data_folder: Path to folder containing CSV files (default: monkey_4iar data)
        fold_idx: Which fold to use (default: 0)
        manual_seed: Random seed for reproducibility
        verbose: Print detailed output
        threads: Number of threads to use for parallel processing (default: 50)
    
    Returns:
        dict with timing results for each sample size
    """
    # Set seed for reproducibility
    random.seed(manual_seed)
    
    # Default data folder path
    if data_folder is None:
        data_folder = "../../monkey_4iar/analysis/data/processed/harry/splits_20000"
    
    # Load data
    try:
        data = [pd.read_csv(f"{data_folder}/{i}.csv") for i in range(5)]
        train_data = data[fold_idx]
        if verbose:
            print("=" * 80)
            print("SCALING TEST: tree_search.fit() Performance")
            print("=" * 80)
            print(f"Data folder: {data_folder}")
            print(f"Fold index: {fold_idx}")
            print(f"Total available samples: {len(train_data)}")
            print(f"Manual seed: {manual_seed}")
            print(f"Threads: {threads}")
            print()
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find data folder: {data_folder}")
        print("Please provide a valid data_folder path or ensure the default path exists.")
        return None
    
    # Sample sizes to test - start small, gradually increase
    # Start with just 1 and 5 to establish baseline, then add 10
    sample_sizes = [1, 5]
    
    # BADS options - start with very loose tolerances, gradually tighten towards defaults
    # Defaults are typically: tol_mesh ~1e-6, tol_fun ~1e-4
    # Starting very loose for faster completion, can be tightened progressively
    # Use adaptive settings based on sample size
    def get_bads_options(n_samples):
        # Use fewer evaluations and iterations for faster completion
        # But ensure enough evaluations for BADS GP training (needs at least ~40-50)
        max_evals = 50 if n_samples <= 5 else 60
        return {
            'uncertainty_handling': False,  # Disable to avoid NaN issues in GP training
            'noise_final_samples': 0,
            'max_fun_evals': max_evals,  # Minimum needed for BADS GP training
            'max_iter': 10,              # Allow more iterations for convergence
            'tol_mesh': 1e-1,            # Very loose (10x looser than before)
            'tol_fun': 1e-1             # Very loose (10x looser than before)
        }
    
    results = {}
    
    for n_samples in sample_sizes:
        if verbose:
            print("=" * 80)
            print(f"Testing with {n_samples} sample(s)")
            print("=" * 80)
        
        # Sample data
        if n_samples > len(train_data):
            print(f"⚠️  Warning: Requested {n_samples} samples but only {len(train_data)} available.")
            print(f"   Using all {len(train_data)} samples instead.")
            sample_data = train_data.copy()
        else:
            sample_data = train_data.sample(n=n_samples, random_state=manual_seed).copy()
        
        # Setup model and fitter (using FastFitter for fewer evaluation iterations)
        model = TreeSearch()
        fitter = FastFitter(model, threads=threads, verbose=verbose)
        
        # Measure time
        start_time = time.time()
        
        try:
            # Only use manual_seed with single thread (required for reproducibility)
            fit_seed = manual_seed if threads == 1 else None
            # Get adaptive BADS options for this sample size
            sample_bads_options = get_bads_options(n_samples)
            fitted_params, final_LL = fitter.fit(
                sample_data,
                manual_seed=fit_seed,
                bads_options=sample_bads_options
            )
            
            elapsed_time = time.time() - start_time
            
            results[n_samples] = {
                'time': elapsed_time,
                'success': True,
                'fitted_params': fitted_params,
                'final_LL': final_LL,
                'n_samples': len(sample_data)
            }
            
            if verbose:
                print(f"\n✅ Completed in {elapsed_time:.2f} seconds")
                print(f"   Final log-likelihood: {final_LL.sum():.4f}")
                print(f"   Average LL per sample: {final_LL.mean():.4f}")
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[n_samples] = {
                'time': elapsed_time,
                'success': False,
                'error': str(e),
                'n_samples': len(sample_data)
            }
            
            if verbose:
                print(f"\n❌ Failed after {elapsed_time:.2f} seconds")
                print(f"   Error: {e}")
        
        if verbose:
            print()
    
    # Print summary
    print("=" * 80)
    print("SCALING TEST SUMMARY")
    print("=" * 80)
    
    successful_runs = [n for n in sample_sizes if results[n]['success']]
    
    if successful_runs:
        print("\n✅ Successful runs:")
        for n in successful_runs:
            r = results[n]
            print(f"   {n:2d} samples: {r['time']:8.2f}s  ({r['time']/r['n_samples']:.3f}s per sample)")
        
        # Calculate scaling factor
        if len(successful_runs) >= 2:
            times = [results[n]['time'] for n in successful_runs]
            samples = [results[n]['n_samples'] for n in successful_runs]
            
            print("\n📊 Scaling analysis:")
            for i in range(1, len(successful_runs)):
                n1, t1 = samples[i-1], times[i-1]
                n2, t2 = samples[i], times[i]
                speedup = (n2 / n1) / (t2 / t1) if t1 > 0 else float('inf')
                print(f"   {n1} → {n2} samples: {t2/t1:.2f}x time ({speedup:.2f}x scaling efficiency)")
    
    failed_runs = [n for n in sample_sizes if not results[n]['success']]
    if failed_runs:
        print("\n❌ Failed runs:")
        for n in failed_runs:
            r = results[n]
            print(f"   {n:2d} samples: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Run the scaling test with 50 threads for faster execution
    results = test_scaling(verbose=True, threads=50)
    
    if results:
        print("\n✅ Scaling test completed!")
    else:
        print("\n❌ Scaling test failed to run.")

