"""
Tests for tree_search_parallel: equivalence with SingleThreadedFitter (n_workers=1)
and timing comparison between original and parallel versions.
"""
import sys
import random
import time
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tree_search import TreeSearch, SingleThreadedFitter
from tree_search_parallel import ParallelFitter


def find_sample_data():
    """Find sample data for testing."""
    candidates = [
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/splits",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-02-20",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-03-13",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2024-02-14",
    ]
    for base in candidates:
        if os.path.exists(base):
            files = glob.glob(f"{base}/split_*.csv")
            if files:
                return sorted(files)[0]
    return None


def load_sample_data(n_trials=30, data_path=None):
    """Load sample data for tests."""
    if data_path is None:
        data_path = find_sample_data()
    if data_path is None:
        raise FileNotFoundError("No sample data found. Expected split_*.csv in monkey_4iar paths.")
    df = pd.read_csv(data_path)
    return df[:n_trials]


def make_model_and_params():
    """Create model and test params for both fitters."""
    templates = {
        "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        "4IAR": [[1, 1, 1, 1]],
    }
    weights = {"2IAR_CON": 1.0, "2IAR_DIS": 0.4, "3IAR": 3.5, "4IAR": 9.0}
    model = TreeSearch(templates=templates, initial_weights=weights, verbose=False)
    params = model.initial_params.copy()
    return model, params


def test_equivalence_n_workers_1(n_trials=20, n_repeats=5, n_iterations=10, manual_seed=42, verbose=True):
    """
    Test that ParallelFitter with n_workers=1 produces EXACTLY the same results
    as SingleThreadedFitter across multiple evaluate() calls.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    data = load_sample_data(n_trials=n_trials)
    model_st, params = make_model_and_params()
    model_pl, _ = make_model_and_params()

    fitter_st = SingleThreadedFitter(model_st, n_repeats=n_repeats, verbose=False)
    fitter_pl = ParallelFitter(model_pl, n_repeats=n_repeats, verbose=False, n_workers=1)

    fitter_st.data = data
    fitter_pl.data = data

    all_match = True
    for i in range(n_iterations):
        random.seed(manual_seed + i)
        res_st = fitter_st.evaluate(params, data)

        random.seed(manual_seed + i)
        res_pl = fitter_pl.evaluate(params, data)

        diff = np.abs(res_st - res_pl)
        if not np.allclose(res_st, res_pl, rtol=0, atol=1e-9):
            all_match = False
            if verbose:
                print(f"Iteration {i}: MISMATCH")
                print(f"  SingleThreaded: {res_st}")
                print(f"  Parallel(n=1):  {res_pl}")
                print(f"  Max diff: {diff.max()}")
        elif verbose and i < 3:
            print(f"Iteration {i}: match (sum={res_st.sum():.6f})")

    if verbose:
        print()
        if all_match:
            print("✅ EQUIVALENCE TEST PASSED: ParallelFitter(n_workers=1) matches SingleThreadedFitter exactly")
        else:
            print("❌ EQUIVALENCE TEST FAILED: Results differ")
    return all_match


def test_timing(n_trials=50, n_repeats=5, n_workers_list=(1, 4, 8), manual_seed=42, verbose=True):
    """
    Compare timing of SingleThreadedFitter vs ParallelFitter for n_repeats=5.
    """
    from multiprocessing import set_start_method
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    random.seed(manual_seed)
    data = load_sample_data(n_trials=n_trials)

    model_st, params = make_model_and_params()
    model_pl, _ = make_model_and_params()

    fitter_st = SingleThreadedFitter(model_st, n_repeats=n_repeats, verbose=False)
    fitter_pl = ParallelFitter(model_pl, n_repeats=n_repeats, verbose=False)

    fitter_st.data = data
    fitter_pl.data = data

    # Warm up
    _ = fitter_st.evaluate(params, data)
    _ = fitter_pl.evaluate(params, data)

    # Time SingleThreadedFitter
    random.seed(manual_seed)
    t0 = time.perf_counter()
    for _ in range(3):
        random.seed(manual_seed)
        _ = fitter_st.evaluate(params, data)
    t_st = (time.perf_counter() - t0) / 3

    results = {"SingleThreaded": t_st}

    for nw in n_workers_list:
        fitter_pl.n_workers = nw
        random.seed(manual_seed)
        t0 = time.perf_counter()
        for _ in range(3):
            random.seed(manual_seed)
            _ = fitter_pl.evaluate(params, data)
        t_pl = (time.perf_counter() - t0) / 3
        results[f"Parallel(n={nw})"] = t_pl

    if verbose:
        print("=" * 60)
        print("TIMING TEST (n_repeats=5, sample dataset)")
        print("=" * 60)
        print(f"n_trials: {n_trials}, n_repeats: {n_repeats}")
        print()
        baseline = results["SingleThreaded"]
        print(f"{'Implementation':<25} {'Time (s)':>12} {'Speedup':>10}")
        print("-" * 50)
        for name, t in results.items():
            speedup = baseline / t if t > 0 else 0
            print(f"{name:<25} {t:>12.4f} {speedup:>10.2f}x")
        print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Trials for timing test")
    parser.add_argument("--skip-timing", action="store_true", help="Skip timing test")
    args = parser.parse_args()

    data_path = find_sample_data()
    if data_path is None:
        print("⚠️  No sample data found. Create synthetic data or set DATA_PATH.")
        print("   Skipping tests that require data.")
        sys.exit(0)

    print("\n--- Equivalence test (n_workers=1 vs SingleThreadedFitter) ---")
    ok = test_equivalence_n_workers_1(verbose=True)
    if not ok:
        sys.exit(1)

    if not args.skip_timing:
        print("\n--- Timing test (n_repeats=5) ---")
        test_timing(n_trials=args.n_trials, verbose=True)

    print("\n✅ All tests completed.")
