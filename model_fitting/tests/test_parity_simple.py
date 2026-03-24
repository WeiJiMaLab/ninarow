import sys
import os
import pandas as pd
import numpy as np
import random
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tree_search import TreeSearch, SingleThreadedFitter
from tree_search_parallel import MultiThreadedFitter

def run_parity_test():
    print("Testing parity between MultiThreadedFitter(n_workers=1) and SingleThreadedFitter...")
    
    # Create a small dataset
    data = pd.DataFrame({
        'black': [64, 12886018120, 2048, 0],
        'white': [0, 20983844, 134217728, 0],
        'color': ['White', 'White', 'Black', 'Black'],
        'move': [4194304, 2147483648, 16384, 8388608]
    })
    
    model = TreeSearch(verbose=False)
    params = model.initial_params
    
    # 1. Run SingleThreadedFitter
    # We must seed random once before each evaluate to ensure they start from same state
    random.seed(42)
    np.random.seed(42)
    single_fitter = SingleThreadedFitter(model, n_repeats=10, verbose=False)
    # evaluation 1
    res_single = single_fitter.evaluate(params, data)
    
    # 2. Run MultiThreadedFitter with n_workers=1
    random.seed(42)
    np.random.seed(42)
    # Create a fresh model copy to be fair
    model2 = TreeSearch(verbose=False)
    multi_fitter = MultiThreadedFitter(model2, n_repeats=10, verbose=False, n_workers=1)
    res_multi = multi_fitter.evaluate(params, data)
    
    print(f"SingleThreaded Results: {res_single}")
    print(f"MultiThreaded Results:  {res_multi}")
    
    parity = np.allclose(res_single, res_multi)
    if parity:
        print("✅ SUCCESS: Results match exactly!")
    else:
        print("❌ FAILURE: Results differ!")
        print(f"Difference: {res_single - res_multi}")

    # 3. Test n_workers=2 (should be different due to RNG sequence split, but should still run)
    print("\nTesting MultiThreadedFitter(n_workers=2)...")
    try:
        multi_fitter2 = MultiThreadedFitter(model2, n_repeats=10, verbose=False, n_workers=2)
        res_multi2 = multi_fitter2.evaluate(params, data)
        print(f"MultiThreaded (n=2) Results: {res_multi2}")
        print("✅ SUCCESS: n_workers=2 executed successfully.")
    except Exception as e:
        print(f"❌ FAILURE: n_workers=2 failed with error: {e}")

if __name__ == "__main__":
    run_parity_test()
