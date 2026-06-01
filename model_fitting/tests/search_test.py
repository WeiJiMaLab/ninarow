#!/usr/bin/env python3
import sys
import os
import numpy as np
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourbynine import *
from tree_search import TreeSearch
from tree_search_fitter import SingleThreadedFitter, MultiThreadedFitter

def create_test_board(black_bits, white_bits):
    """Creates a legal board state with given bit patterns."""
    return fourbynine_board(fourbynine_pattern(black_bits), fourbynine_pattern(white_bits))

def generate_dummy_data(n_trials=10):
    """Generate minimal dummy data for testing likelihoods."""
    import pandas as pd
    data = []
    for i in range(n_trials):
        data.append({
            "black": 0,
            "white": 0,
            "move": 1 << 13, # Move at position 13
            "color": "black"
        })
    return pd.DataFrame(data)

def test_fitter_reproducibility():
    """Verify that fitters produce consistent results when seeded."""
    print("\n--- Testing Fitter Reproducibility ---")
    
    data = generate_dummy_data()
    model = TreeSearch(verbose=False)
    params = [p["initial_value"] for p in model.parameter_list]
    
    # Use SingleThreadedFitter for simplicity in verification
    fitter = SingleThreadedFitter(model, n_repeats=10)
    
    random.seed(42)
    ll1 = fitter.evaluate(params, data)[0].sum()
    
    random.seed(42)
    ll2 = fitter.evaluate(params, data)[0].sum()
    
    match = abs(ll1 - ll2) < 1e-10
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{status}: Seeded Reproducibility")
    print(f"      Run 1: {ll1}, Run 2: {ll2}")
    
    return match

def test_search_determinism():
    """Verify that search results are deterministic when noise is disabled."""
    print("\n--- Testing Search Determinism ---")
    
    model = TreeSearch(verbose=False)
    # [pruning, stopping, feature_drop, lapse, opp_scale, center_weight]
    params = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0] + [1.0]*len(model.sorted_groups)
    model.set_params(params)
    model.heuristic.set_noise_enabled(False)
    
    board = create_test_board(0, 0)
    
    # Run search twice
    move1 = model.predict(board)
    move2 = model.predict(board)
    
    match = (move1 == move2)
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{status}: Search Determinism (Move 1: {move1}, Move 2: {move2})")
    
    return match

def main():
    print("=" * 60)
    print("TREE SEARCH SYSTEM TESTS")
    print("=" * 60)
    
    results = [
        test_fitter_reproducibility(),
        test_search_determinism()
    ]
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
