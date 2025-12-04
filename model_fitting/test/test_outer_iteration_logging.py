"""
Test script to verify that outer iteration count is logged correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import glob
from tree_search import TreeSearch, SingleThreadedFitter
import random

def load_test_data(data_folder, n_trials=5):
    """Load test data from CSV files like other test files."""
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    n_splits = len(split_files)
    if n_splits == 0:
        raise ValueError(f"No split files found in {data_folder}")
    data = [pd.read_csv(f"{data_folder}/split_{i}.csv") for i in range(n_splits)]
    # Use first split, limited to n_trials
    train_data = data[0][:n_trials]
    return train_data

def test_outer_iteration_logging():
    """Test that outer iteration count is logged."""
    print("=" * 80)
    print("Testing Outer Iteration Logging")
    print("=" * 80)
    
    # Load test data from an available folder
    data_folder = "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/models/2023-02-20"
    data = load_test_data(data_folder, n_trials=20)
    print(f"Loaded {len(data)} rows of data from {data_folder}")
    data["expected_counts"] = 1
    
    # Create model and fitter
    model = TreeSearch()
    fitter = SingleThreadedFitter(model, verbose=True, train_repeats=1)
    
    # Set data
    fitter.data = data
    
    # Run BADS with limited iterations to see the logging
    print("\nRunning BADS optimization (limited to 20 function evaluations)...")
    print("Look for '[OuterIter-X]' in the output:\n")
    
    bads_options = {
        'uncertainty_handling': True,
        'noise_final_samples': 0,
        'max_fun_evals': 100,  # Limit to 20 evaluations for quick test
    }
    
    try:
        fitted_params, final_LL = fitter.fit(
            data=data,
            manual_seed=1,
            bads_options=bads_options
        )
        print("\n✅ Test completed successfully!")
        print(f"Final parameters: {fitted_params}")
        print(f"Final LL: {final_LL}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_outer_iteration_logging()

