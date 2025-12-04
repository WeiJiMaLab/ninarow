"""
Verification that tree_search, tree_search_parallel, and model_fit implementations produce identical NLL values.

This verifies that all three implementations produce the same results
for the first N optimization iterations.
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import random
from tree_search import TreeSearch, Fitter, initialize_thread_pool, SingleThreadedFitter
from model_fit import DefaultModel, ModelFitter
from parsers import CSVMove
from fourbynine import fourbynine_board, fourbynine_pattern, fourbynine_move
from prodict import Prodict
import model_fit
import time


def parse_monkey_4iar_dataframe(df):
    """
    Convert a pandas DataFrame from monkey_4iar format (read_fold_csv) 
    to a list of CSVMove objects for use with model_fit.py
    """
    required_cols = ['black', 'white', 'move', 'color']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    moves = []
    for idx, row in df.iterrows():
        black_pieces = int(row['black'])
        white_pieces = int(row['white'])
        board = fourbynine_board(
            fourbynine_pattern(black_pieces), 
            fourbynine_pattern(white_pieces)
        )
        move_bitfield = int(row['move'])
        move_index = move_bitfield.bit_length() - 1
        move_obj = fourbynine_move(move_index, 0.0, board.active_player())
        csv_move = CSVMove(
            board=board,
            move=move_obj,
            time=0,
            group_id=1,
            participant_id=1,
            unique_id=row.game_id.replace(",", ".")
        )
        moves.append(csv_move)
    return moves


def verify_implementations(data_folder, fold_idx=0, n_trials=5, cutoff=1.2, 
                          manual_seed=1, n_iterations=10, verbose=True, feature_drop=0.0):
    """
    Verify that tree_search, tree_search_parallel (MultiThreadedFitter), and model_fit 
    produce identical NLL values.
    
    Returns dict with comparison results including match status and NLL values.
    """
    # Set seed
    random.seed(manual_seed)
    
    # Load data - only load splits that exist
    import glob as _glob
    split_files = sorted(_glob.glob(f"{data_folder}/split_*.csv"))
    n_splits = len(split_files)
    if n_splits == 0:
        raise ValueError(f"No split files found in {data_folder}")
    data = [pd.read_csv(f"{data_folder}/split_{i}.csv") for i in range(n_splits)]
    fold_idx = min(fold_idx, n_splits - 1)  # Ensure fold_idx is valid
    train_data = data[fold_idx][:n_trials]
    
    if verbose:
        print("=" * 80)
        print("VERIFICATION: Comparing tree_search vs tree_search_parallel vs model_fit")
        print("=" * 80)
        print(f"Data folder: {data_folder}")
        print(f"Fold index: {fold_idx}")
        print(f"Number of trials: {n_trials}")
        print(f"Cutoff: {cutoff}")
        print(f"Feature drop: {feature_drop} (0 = disabled)")
        print(f"Manual seed: {manual_seed}")
        print(f"Checking first {n_iterations} iterations")
        print()

    templates = {
        "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        "4IAR": [[1, 1, 1, 1]],
    }

    weights = {
        "2IAR_CON": 1.0,
        "2IAR_DIS": 0.4,
        "3IAR": 3.5,
        "4IAR": 9.0,
    }
    
    # ===== Setup tree_search (parallel Fitter) =====
    treesearch = TreeSearch(templates=templates, initial_weights=weights)
    treesearch.cutoff = cutoff
    if 'feature_drop' in treesearch.param_names:
        feature_drop_idx = treesearch.param_names.index('feature_drop')
        treesearch.initial_params[feature_drop_idx] = feature_drop
        treesearch.lower_bound[feature_drop_idx] = feature_drop
        treesearch.upper_bound[feature_drop_idx] = feature_drop
    fitter = Fitter(treesearch, threads=1, verbose=False)
    
    # ===== Setup tree_search (SingleThreadedFitter) =====
    treesearch_st = TreeSearch(templates=templates, initial_weights=weights)
    treesearch_st.cutoff = cutoff
    if 'feature_drop' in treesearch_st.param_names:
        feature_drop_idx = treesearch_st.param_names.index('feature_drop')
        treesearch_st.initial_params[feature_drop_idx] = feature_drop
        treesearch_st.lower_bound[feature_drop_idx] = feature_drop
        treesearch_st.upper_bound[feature_drop_idx] = feature_drop
    fitter_st = SingleThreadedFitter(treesearch_st, verbose=False, train_repeats=1)
    
    # ===== Setup model_fit (original implementation) =====
    defaultmodel = DefaultModel()
    defaultmodel.cutoff = cutoff
    defaultmodel.x0 = [treesearch.initial_params[i] for i in range(len(treesearch.initial_params))]
    defaultmodel.ub = [treesearch.upper_bound[i] for i in range(len(treesearch.upper_bound))]
    defaultmodel.lb = [treesearch.lower_bound[i] for i in range(len(treesearch.lower_bound))]
    defaultmodel.pub = [treesearch.plausible_upper_bound[i] for i in range(len(treesearch.plausible_upper_bound))]
    defaultmodel.plb = [treesearch.plausible_lower_bound[i] for i in range(len(treesearch.plausible_lower_bound))]
    model_fitter = ModelFitter(
        args=Prodict({'threads': 1, 'random_sample': False, 'verbose': False}), 
        model=defaultmodel
    )
    
    # Verify initial parameters match
    assert np.allclose(treesearch.initial_params, defaultmodel.x0), "Initial params don't match!"
    assert np.allclose(treesearch.initial_params, treesearch_st.initial_params), "Initial params don't match between tree_search variants!"
    
    test_params = treesearch.initial_params
    
    # ===== TREE_SEARCH (Parallel Fitter) =====
    if verbose:
        print("Running tree_search (Fitter) setup...")
    
    # Reset random state and reinitialize thread pool
    random.seed(manual_seed)
    initialize_thread_pool(1, manual_seed=manual_seed)
    
    fitter.data = train_data.copy()
    fitter.data["expected_counts"] = 1
    fitter.iteration_count = 0  # Reset iteration count
    
    nll_tree_search = []
    tree_search_times = []
    
    for i in range(n_iterations):
        start_time = time.perf_counter()
        nll = fitter.optimize(test_params)
        elapsed_time = time.perf_counter() - start_time
        tree_search_times.append(elapsed_time)
        nll_tree_search.append(nll)
    
    if verbose:
        print(f"  Completed {n_iterations} iterations")
    
    # ===== TREE_SEARCH (SingleThreadedFitter) =====
    if verbose:
        print("\nRunning tree_search (SingleThreadedFitter) setup...")
    
    fitter_st.data = train_data.copy()
    fitter_st.data["expected_counts"] = 1
    fitter_st.iteration_count = 0  # Reset iteration count
    
    nll_single_threaded = []
    single_threaded_times = []
    
    # Reset random state right before optimization loop to match tree_search worker state
    # The Pool worker does random.seed(1) then random.randint in set_seeds, so we need to match
    random.seed(manual_seed)
    _ = random.randint(0, 2**64)  # Consume one random number to match worker state
    
    for i in range(n_iterations):
        start_time = time.perf_counter()
        nll = fitter_st.optimize(test_params)
        elapsed_time = time.perf_counter() - start_time
        single_threaded_times.append(elapsed_time)
        nll_single_threaded.append(nll)
    
    if verbose:
        print(f"  Completed {n_iterations} iterations")
    
    # ===== MODEL_FIT =====
    if verbose:
        print("\nRunning model_fit setup...")
    
    # Reset random state and reinitialize thread pool
    random.seed(manual_seed)
    model_fit.initialize_thread_pool(1, manual_seed=manual_seed)
    
    moves_mf = parse_monkey_4iar_dataframe(train_data)
    
    move_tasks_mf = {}
    for move in moves_mf:
        move_tasks_mf[move] = model_fit.SuccessFrequencyTracker(defaultmodel.expt_factor)
        move_tasks_mf[move].required_success_count = 1
    
    nll_model_fit = []
    model_fit_times = []
    
    def opt_fun(x):
        start_time = time.perf_counter()
        loglik = sum(list(model_fitter.compute_loglik(move_tasks_mf, x).values()))
        elapsed_time = time.perf_counter() - start_time
        model_fit_times.append(elapsed_time)
        nll_model_fit.append(loglik)
        return loglik
    
    for i in range(n_iterations):
        opt_fun(test_params)
    
    if verbose:
        print(f"  Completed {n_iterations} iterations")
    
    # ===== COMPARISON =====
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
    
    tolerance = 1e-5
    all_match_ts_mf = True
    all_match_st_mf = True
    all_match_ts_st = True
    
    if verbose:
        print(f"\nComparing {n_iterations} iterations (tolerance: {tolerance})")
        print(f"{'Iter':>4} | {'tree_search':>12} | {'tree_search(ST)':>12} | {'model_fit':>12} | {'TS-MF diff':>12} | {'ST-MF diff':>12} | {'TS-ST diff':>12}")
        print("-" * 95)
    
    for i in range(n_iterations):
        ts_nll = nll_tree_search[i]
        st_nll = nll_single_threaded[i]
        mf_nll = nll_model_fit[i]
        
        diff_ts_mf = abs(ts_nll - mf_nll)
        diff_st_mf = abs(st_nll - mf_nll)
        diff_ts_st = abs(ts_nll - st_nll)
        
        if diff_ts_mf > tolerance:
            all_match_ts_mf = False
        if diff_st_mf > tolerance:
            all_match_st_mf = False
        if diff_ts_st > tolerance:
            all_match_ts_st = False
        
        if verbose:
            print(f"{i:>4} | {ts_nll:>12.6f} | {st_nll:>12.6f} | {mf_nll:>12.6f} | {diff_ts_mf:>12.2e} | {diff_st_mf:>12.2e} | {diff_ts_st:>12.2e}")
    
    all_match = all_match_ts_mf and all_match_st_mf and all_match_ts_st
    
    if verbose:
        print()
        if all_match:
            print("✅ SUCCESS: All NLL values match across all three implementations!")
        else:
            print("❌ FAILURE: Some NLL values differ!")
            if not all_match_ts_mf:
                print("  ❌ tree_search vs model_fit mismatch")
            if not all_match_st_mf:
                print("  ❌ tree_search (SingleThread) vs model_fit mismatch")
            if not all_match_ts_st:
                print("  ❌ tree_search (Fitter) vs tree_search (SingleThread) mismatch")
        
        # Timing summary
        print("\n" + "=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        
        avg_ts = np.mean(tree_search_times)
        avg_st = np.mean(single_threaded_times)
        avg_mf = np.mean(model_fit_times)
        
        print(f"\n{'Implementation':<25} {'Avg Time (ms)':>15} {'Total Time (s)':>15}")
        print("-" * 60)
        print(f"{'tree_search (Fitter)':<25} {avg_ts*1000:>15.2f} {np.sum(tree_search_times):>15.4f}")
        print(f"{'tree_search (ST)':<25} {avg_st*1000:>15.2f} {np.sum(single_threaded_times):>15.4f}")
        print(f"{'model_fit':<25} {avg_mf*1000:>15.2f} {np.sum(model_fit_times):>15.4f}")
        
        print(f"\n📊 Speedup Analysis (relative to model_fit):")
        print(f"   tree_search (Fitter):        {avg_mf/avg_ts:.2f}x")
        print(f"   tree_search (SingleThread):  {avg_mf/avg_st:.2f}x")
        
        print(f"\n📊 Comparison (tree_search (SingleThread) vs tree_search (Fitter)):")
        if avg_st < avg_ts:
            print(f"   tree_search (SingleThread) is {avg_ts/avg_st:.2f}x faster")
        else:
            print(f"   tree_search (Fitter) is {avg_st/avg_ts:.2f}x faster")
    
    return {
        'all_match': all_match,
        'match_ts_mf': all_match_ts_mf,
        'match_st_mf': all_match_st_mf,
        'match_ts_st': all_match_ts_st,
        'nll_tree_search': nll_tree_search,
        'nll_single_threaded': nll_single_threaded,
        'nll_model_fit': nll_model_fit,
        'tree_search_times': tree_search_times,
        'single_threaded_times': single_threaded_times,
        'model_fit_times': model_fit_times,
        'avg_tree_search_time': np.mean(tree_search_times),
        'avg_single_threaded_time': np.mean(single_threaded_times),
        'avg_model_fit_time': np.mean(model_fit_times)
    }


if __name__ == "__main__":
    # Run verification
    data_folder = "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/models/2023-02-20"
    result = verify_implementations(
        data_folder=data_folder,
        fold_idx=0,
        n_trials=5,
        cutoff=100.0,  # Large cutoff - no early termination
        manual_seed=1,
        n_iterations=5,
        verbose=True,
        feature_drop=0.0
    )
    
    # Print final summary
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)
    
    if result['all_match']:
        print('✅✅✅ ALL CHECKS PASSED!')
        print('\n✅ All three implementations produce identical NLL values')
        print('   - tree_search (Fitter)')
        print('   - tree_search (SingleThread)')
        print('   - model_fit')
    else:
        print('❌ SOME CHECKS FAILED:')
        if not result['match_ts_mf']:
            print('  ❌ tree_search vs model_fit mismatch')
        if not result['match_st_mf']:
            print('  ❌ tree_search (SingleThread) vs model_fit mismatch')
        if not result['match_ts_st']:
            print('  ❌ tree_search (Fitter) vs tree_search (SingleThread) mismatch')
    
    print('\n' + '=' * 80)
    print('PERFORMANCE RANKING')
    print('=' * 80)
    
    times = [
        ('tree_search (Fitter)', result['avg_tree_search_time']),
        ('tree_search (SingleThread)', result['avg_single_threaded_time']),
        ('model_fit', result['avg_model_fit_time'])
    ]
    times.sort(key=lambda x: x[1])
    
    print(f"\nFastest to slowest:")
    for i, (name, t) in enumerate(times):
        speedup = times[-1][1] / t  # relative to slowest
        print(f"  {i+1}. {name:<25} {t*1000:>8.2f}ms  ({speedup:.2f}x vs slowest)")
    
    print('=' * 80)
    
    # Exit with appropriate code
    sys.exit(0 if result['all_match'] else 1)
