"""
Verify that SingleThreadedFitter and model_fit produce identical NLL values.

This test compares the two implementations across multiple optimization iterations
to ensure they produce the same results.
"""
import sys
from pathlib import Path
import random
import time
import glob

import numpy as np
import pandas as pd
from prodict import Prodict

sys.path.insert(0, str(Path(__file__).parent.parent))
from tree_search import TreeSearch, SingleThreadedFitter
from model_fit import DefaultModel, ModelFitter
import model_fit
from parsers import CSVMove
from fourbynine import fourbynine_board, fourbynine_pattern, fourbynine_move


def load_data(data_folder, fold_idx=0, n_trials=20):
    """Load test data from split CSV files."""
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    if not split_files:
        raise ValueError(f"No split files found in {data_folder}")
    
    data = [pd.read_csv(f) for f in split_files]
    fold_idx = min(fold_idx, len(data) - 1)
    return data[fold_idx][:n_trials]


def parse_dataframe_to_moves(df):
    """Convert DataFrame to list of CSVMove objects for model_fit."""
    required_cols = ['black', 'white', 'move', 'color', 'trial_id']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    moves = []
    for _, row in df.iterrows():
        board = fourbynine_board(
            fourbynine_pattern(int(row['black'])), 
            fourbynine_pattern(int(row['white']))
        )
        move_bitfield = int(row['move'])
        move_index = move_bitfield.bit_length() - 1
        move_obj = fourbynine_move(move_index, 0.0, board.active_player())
        
        moves.append(CSVMove(
            board=board,
            move=move_obj,
            time=0,
            group_id=1,
            participant_id=1,
            unique_id=str(row.trial_id).replace(",", ".")
        ))
    return moves


def setup_models(cutoff=100.0, feature_drop=0.0):
    """Setup TreeSearch and DefaultModel with matching parameters."""
    templates = {
        "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        "4IAR": [[1, 1, 1, 1]],
    }
    weights = {"2IAR_CON": 1.0, "2IAR_DIS": 0.4, "3IAR": 3.5, "4IAR": 9.0}
    
    # Setup TreeSearch
    model_st = TreeSearch(templates=templates, initial_weights=weights)
    model_st.cutoff = cutoff
    
    if 'feature_drop' in model_st.param_names:
        idx = model_st.param_names.index('feature_drop')
        model_st.initial_params[idx] = feature_drop
        model_st.lower_bound[idx] = feature_drop
        model_st.upper_bound[idx] = feature_drop
    
    # Setup DefaultModel to match
    model_mf = DefaultModel()
    model_mf.cutoff = cutoff
    model_mf.x0 = model_st.initial_params.copy()
    model_mf.ub = model_st.upper_bound.copy()
    model_mf.lb = model_st.lower_bound.copy()
    model_mf.pub = model_st.plausible_upper_bound.copy()
    model_mf.plb = model_st.plausible_lower_bound.copy()
    
    assert np.allclose(model_st.initial_params, model_mf.x0), "Initial params mismatch!"
    
    return model_st, model_mf


def run_tree_search_optimization(fitter, params, n_iterations, manual_seed):
    """Run optimization iterations with SingleThreadedFitter."""
    random.seed(manual_seed)
    _ = random.randint(0, 2**64)  # Match worker state
    
    nlls, times = [], []
    for _ in range(n_iterations):
        start = time.perf_counter()
        nll = fitter.optimize(params)
        times.append(time.perf_counter() - start)
        nlls.append(nll)
    return nlls, times


def run_model_fit_optimization(fitter, move_tasks, params, n_iterations, manual_seed):
    """Run optimization iterations with model_fit."""
    random.seed(manual_seed)
    model_fit.initialize_thread_pool(1, manual_seed=manual_seed)
    
    nlls, times = [], []
    for _ in range(n_iterations):
        start = time.perf_counter()
        loglik = sum(fitter.compute_loglik(move_tasks, params).values())
        times.append(time.perf_counter() - start)
        nlls.append(loglik)
    return nlls, times


def compare_results(nll_st, nll_mf, tolerance=1e-5, verbose=True):
    """Compare NLL values from both implementations."""
    matches = []
    for i, (st_val, mf_val) in enumerate(zip(nll_st, nll_mf)):
        diff = abs(st_val - mf_val)
        matches.append(diff <= tolerance)
        
        if verbose:
            status = "✓" if diff <= tolerance else "✗"
            print(f"{i:>4} | {st_val:>12.6f} | {mf_val:>12.6f} | {diff:>12.2e} {status}")
    
    all_match = all(matches)
    
    if verbose:
        print()
        if all_match:
            print("✅ SUCCESS: All NLL values match!")
        else:
            print(f"❌ FAILURE: {sum(matches)}/{len(matches)} iterations match")
    
    return all_match


def verify_implementations(data_folder, fold_idx=0, n_trials=20, cutoff=100.0,
                          manual_seed=1, n_iterations=20, verbose=True, feature_drop=0.0):
    """Verify that both implementations produce identical NLL values."""
    
    if verbose:
        print("=" * 80)
        print("VERIFICATION: SingleThreadedFitter vs model_fit")
        print("=" * 80)
        print(f"Data folder: {data_folder}")
        print(f"Trials: {n_trials}, Iterations: {n_iterations}, Seed: {manual_seed}")
        print()
    
    # Load data
    data = load_data(data_folder, fold_idx, n_trials)
    data["expected_counts"] = 1
    
    # Setup models
    model_st, model_mf = setup_models(cutoff, feature_drop)
    fitter_st = SingleThreadedFitter(model_st, verbose=False)
    fitter_mf = ModelFitter(
        args=Prodict({'threads': 1, 'random_sample': False, 'verbose': False}),
        model=model_mf
    )
    
    # Prepare data for model_fit
    moves = parse_dataframe_to_moves(data)
    move_tasks = {
        move: model_fit.SuccessFrequencyTracker(model_mf.expt_factor)
        for move in moves
    }
    for task in move_tasks.values():
        task.required_success_count = 1
    
    # Setup fitter_st
    fitter_st.data = data.copy()
    fitter_st.iteration_count = 0
    
    test_params = model_st.initial_params.copy()
    
    # Run optimizations
    if verbose:
        print("Running SingleThreadedFitter...")
    nll_st, times_st = run_tree_search_optimization(
        fitter_st, test_params, n_iterations, manual_seed
    )
    
    if verbose:
        print("Running model_fit...")
    nll_mf, times_mf = run_model_fit_optimization(
        fitter_mf, move_tasks, test_params, n_iterations, manual_seed
    )
    
    # Compare results
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"{'Iter':>4} | {'SingleThread':>12} | {'model_fit':>12} | {'Difference':>12} | {'Status'}")
        print("-" * 60)
    
    all_match = compare_results(nll_st, nll_mf, verbose=verbose)
    
    # Timing summary
    if verbose:
        avg_st = np.mean(times_st)
        avg_mf = np.mean(times_mf)
        
        print("\n" + "=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        print(f"{'Implementation':<25} {'Avg Time (ms)':>15} {'Total Time (s)':>15}")
        print("-" * 60)
        print(f"{'SingleThreadedFitter':<25} {avg_st*1000:>15.2f} {np.sum(times_st):>15.4f}")
        print(f"{'model_fit':<25} {avg_mf*1000:>15.2f} {np.sum(times_mf):>15.4f}")
        print(f"\nSpeedup: {avg_mf/avg_st:.2f}x")
    
    return {
        'all_match': all_match,
        'nll_single_threaded': nll_st,
        'nll_model_fit': nll_mf,
        'avg_single_threaded_time': np.mean(times_st),
        'avg_model_fit_time': np.mean(times_mf)
    }


if __name__ == "__main__":
    data_folder = "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-02-20"
    
    result = verify_implementations(
        data_folder=data_folder,
        fold_idx=0,
        n_trials=20,
        cutoff=100.0,
        manual_seed=1,
        n_iterations=20,
        verbose=True,
        feature_drop=0.0
    )
    
    # Final summary
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)
    
    if result['all_match']:
        print('✅✅✅ ALL CHECKS PASSED!')
        print('\nBoth implementations produce identical NLL values')
    else:
        print('❌ CHECKS FAILED: Implementations produce different results')
    
    print(f"\nPerformance: SingleThreadedFitter is {result['avg_model_fit_time']/result['avg_single_threaded_time']:.2f}x faster")
    print('=' * 80)
    
    sys.exit(0 if result['all_match'] else 1)
