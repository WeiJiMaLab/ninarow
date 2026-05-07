#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def run_benchmark(n_trials=200):
    print(f"\n--- Running Benchmark (N={n_trials}) ---")
    data = generate_dummy_data(n_trials)
    model = TreeSearch(verbose=False)
    params = [p["initial_value"] for p in model.parameter_list]
    
    # Warmup
    fitter = MultiThreadedFitter(model, n_workers=1)
    fitter.evaluate(params, data)
    
    # Timing
    start_time = time.time()
    fitter.evaluate(params, data)
    elapsed = time.time() - start_time
    
    ms_per_trial = (elapsed / n_trials) * 1000
    print(f"✅ Performance: {ms_per_trial:.2f} ms/trial ({elapsed:.4f} s total)")
    
    return ms_per_trial < 100 # Arbitrary threshold for "sanity"

def main():
    print("=" * 60)
    print("PERFORMANCE TIMING TESTS")
    print("=" * 60)
    
    run_benchmark()
    
    print("=" * 60)

if __name__ == "__main__":
    main()
