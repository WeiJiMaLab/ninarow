#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Add model_fitting/ (sibling of tests/) to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_fitting"))

from tree_search import TreeSearch
from tree_search_fitter import MultiThreadedFitter

def generate_dummy_data(n_trials=100):
    """Generate dummy data for timing benchmarks."""
    import pandas as pd
    data = []
    for i in range(n_trials):
        data.append({
            "black": 0,
            "white": 0,
            "move": 1 << 13,
            "color": "black"
        })
    return pd.DataFrame(data)

def run_benchmark(n_trials=100, worker_counts=[1, 2, 4, 8, 12, 16, 24, 32]):
    print(f"\n--- Scaling Benchmark (N_trials={n_trials}) ---")
    data = generate_dummy_data(n_trials)
    model = TreeSearch(verbose=False)
    params = [p["initial_value"] for p in model.parameter_list]
    
    # Speed up search for timing benchmark
    if "stopping_prob" in model.param_names:
        idx = model.param_names.index("stopping_prob")
        params[idx] = 0.5
    
    print(f"{'Workers':<10} | {'Total Time (s)':<15} | {'ms/trial':<12} | {'Speedup':<10}")
    print("-" * 55)
    
    base_time = None
    
    for n in worker_counts:
        # Check if we have enough CPUs
        if n > os.cpu_count() * 2: 
            continue
            
        fitter = MultiThreadedFitter(model, n_workers=n)
        
        # Warmup
        fitter.evaluate(params, data)
        
        start_time = time.time()
        fitter.evaluate(params, data)
        elapsed = time.time() - start_time
        
        if base_time is None:
            base_time = elapsed
            
        speedup = base_time / elapsed
        ms_per_trial = (elapsed / n_trials) * 1000
        
        print(f"{n:<10} | {elapsed:<15.4f} | {ms_per_trial:<12.2f} | {speedup:<10.2f}x")
    
    return True

def main():
    print("=" * 60)
    print("PERFORMANCE SCALING TESTS")
    print("=" * 60)
    
    run_benchmark()
    
    print("=" * 60)

if __name__ == "__main__":
    main()
