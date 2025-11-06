"""
Verification that tree_search and model_fit implementations produce identical NLL values.

This verifies that the modular TreeSearch implementation produces the same results
as the original model_fit implementation for the first N optimization iterations.
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import random
from tree_search import TreeSearch, Fitter, initialize_thread_pool
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
                          manual_seed=1, n_iterations=10, verbose=True, feature_drop=0.2):
    """
    Verify that tree_search and model_fit produce identical NLL values.
    
    Returns dict with comparison results including match status and NLL values.
    """
    # Set seed
    random.seed(manual_seed)
    
    # Load data
    data = [pd.read_csv(f"{data_folder}/{i}.csv") for i in range(5)]
    train_data = data[fold_idx][:n_trials]
    
    if verbose:
        print("=" * 60)
        print("VERIFICATION: Comparing tree_search vs model_fit")
        print("=" * 60)
        print(f"Data folder: {data_folder}")
        print(f"Fold index: {fold_idx}")
        print(f"Number of trials: {n_trials}")
        print(f"Cutoff: {cutoff}")
        print(f"Feature drop: {feature_drop} (0 = disabled)")
        print(f"Manual seed: {manual_seed}")
        print(f"Checking first {n_iterations} iterations")
        print()
    
    # Setup tree_search (modular implementation)
    treesearch = TreeSearch()
    treesearch.cutoff = cutoff
    # Override feature_drop parameter
    treesearch.param_names = treesearch.param_names.copy()
    if 'feature_drop' in treesearch.param_names:
        feature_drop_idx = treesearch.param_names.index('feature_drop')
        treesearch.initial_params[feature_drop_idx] = feature_drop
        treesearch.lower_bound[feature_drop_idx] = feature_drop
        treesearch.upper_bound[feature_drop_idx] = feature_drop
    fitter = Fitter(treesearch, threads=1, verbose=False)
    
    # Setup model_fit (original implementation)
    defaultmodel = DefaultModel()
    defaultmodel.cutoff = cutoff
    # Override feature_drop parameter by modifying x0 directly
    # Parameter order: [pruning, stopping_prob, feature_drop, lapse, c_opp, w_center, ...]
    if len(defaultmodel.x0) >= 3:  # feature_drop is at index 2
        defaultmodel.x0[2] = feature_drop
        defaultmodel.lb[2] = feature_drop
        defaultmodel.ub[2] = feature_drop
    model_fitter = ModelFitter(
        args=Prodict({'threads': 1, 'random_sample': False, 'verbose': False}), 
        model=defaultmodel
    )
    
    # Initialize pools
    initialize_thread_pool(1, manual_seed=manual_seed)
    model_fit.initialize_thread_pool(1, manual_seed=manual_seed)
    
    # Verify initial parameters match
    assert np.allclose(treesearch.initial_params, defaultmodel.x0), "Initial params don't match!"
    assert np.allclose(treesearch.upper_bound, defaultmodel.ub), "Upper bounds don't match!"
    assert np.allclose(treesearch.lower_bound, defaultmodel.lb), "Lower bounds don't match!"
    
    # ===== TREE_SEARCH APPROACH =====
    if verbose:
        print("Running tree_search.fit() setup...")
    
    # Set up fitter like fit() does
    fitter.data = train_data.copy()
    fitter.data["expected_counts"] = 1
    
    # Do initial evaluation
    initial_LL_ts = fitter.evaluate(treesearch.initial_params, fitter.data)
    fitter.data["expected_counts"] = fitter.calculate_expected_counts(initial_LL_ts, treesearch.c).astype(int)
    expected_counts_ts = fitter.data["expected_counts"].tolist()
    
    if verbose:
        print(f"  Initial LL: {initial_LL_ts}")
        print(f"  Expected counts: {expected_counts_ts}")
    
    # Capture NLL values from optimize calls
    nll_tree_search = []
    tree_search_times = []
    
    # Call optimize a few times with the same params BADS would use
    # We'll use the initial params repeatedly to simulate BADS iterations
    test_params = treesearch.initial_params
    for i in range(n_iterations):
        # Time the optimize call
        start_time = time.perf_counter()
        nll = fitter.optimize(test_params)
        elapsed_time = time.perf_counter() - start_time
        tree_search_times.append(elapsed_time)
        nll_tree_search.append(nll)
    
    # ===== MODEL_FIT APPROACH =====
    if verbose:
        print("\nRunning model_fit.fit_model() setup...")
    
    # Reset random seed
    random.seed(manual_seed)
    
    # Parse data
    moves_mf = parse_monkey_4iar_dataframe(train_data)
    
    # Do initial estimation
    average_l_values = defaultmodel.estimate_initial_l_value_guess(model_fitter, moves_mf)
    counts = model_fitter.generate_attempt_counts(np.array(average_l_values), defaultmodel.c)
    
    if verbose:
        print(f"  Initial L values: {average_l_values}")
        print(f"  Generated counts: {counts.tolist()}")
    
    # Create move_tasks
    move_tasks_mf = {}
    for move in moves_mf:
        move_tasks_mf[move] = model_fit.SuccessFrequencyTracker(defaultmodel.expt_factor)
    for i in range(len(counts)):
        move_tasks_mf[moves_mf[i]].required_success_count = int(counts[i])
    
    required_counts_mf = [task.required_success_count for task in move_tasks_mf.values()]
    
    if verbose:
        print(f"  Required success counts: {required_counts_mf}")
    
    # Check if expected counts match
    expected_counts_match = (expected_counts_ts == required_counts_mf)
    
    if verbose:
        ec_match_str = "✅" if expected_counts_match else "❌"
        print(f"\nExpected counts match: {ec_match_str} {expected_counts_match}")
        if not expected_counts_match:
            print(f"  tree_search: {expected_counts_ts}")
            print(f"  model_fit: {required_counts_mf}")
    
    # Create opt_fun and capture NLL values
    nll_model_fit = []
    model_fit_times = []
    
    def opt_fun(x):
        start_time = time.perf_counter()
        loglik = sum(list(model_fitter.compute_loglik(move_tasks_mf, x).values()))
        elapsed_time = time.perf_counter() - start_time
        model_fit_times.append(elapsed_time)
        nll_model_fit.append(loglik)
        return loglik
    
    # Call opt_fun a few times
    for i in range(n_iterations):
        nll = opt_fun(test_params)
    
    # ===== COMPARISON =====
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
    
    # Compare NLL values
    n_iterations_actual = min(len(nll_tree_search), len(nll_model_fit))
    differences = []
    all_match = True
    tolerance = 1e-5  # Allow small floating point differences
    
    if verbose:
        print(f"\nComparing {n_iterations_actual} iterations (tolerance: {tolerance})")
    
    for i in range(n_iterations_actual):
        ts_nll = nll_tree_search[i]
        mf_nll = nll_model_fit[i]
        diff = abs(ts_nll - mf_nll)
        differences.append(diff)
        
        if diff > tolerance:
            all_match = False
        
        if verbose:
            match_str = "✅" if diff <= tolerance else "❌"
            ts_time = tree_search_times[i] if i < len(tree_search_times) else 0
            mf_time = model_fit_times[i] if i < len(model_fit_times) else 0
            print(f"  Iteration {i}: tree_search={ts_nll:.6f} ({ts_time:.4f}s), model_fit={mf_nll:.6f} ({mf_time:.4f}s), diff={diff:.6e} {match_str}")
    
    if verbose:
        print()
        if all_match:
            print("✅ SUCCESS: All NLL values match!")
        else:
            print("❌ FAILURE: NLL values differ!")
        
        # Print timing summary
        if tree_search_times and model_fit_times:
            print("\n" + "=" * 60)
            print("TIMING SUMMARY")
            print("=" * 60)
            avg_ts_time = np.mean(tree_search_times)
            avg_mf_time = np.mean(model_fit_times)
            total_ts_time = np.sum(tree_search_times)
            total_mf_time = np.sum(model_fit_times)
            
            print(f"Tree Search (tree_search):")
            print(f"  Average time per iteration: {avg_ts_time:.4f}s")
            print(f"  Total time ({n_iterations_actual} iterations): {total_ts_time:.4f}s")
            print(f"\nModel Fit (model_fit):")
            print(f"  Average time per iteration: {avg_mf_time:.4f}s")
            print(f"  Total time ({n_iterations_actual} iterations): {total_mf_time:.4f}s")
            print(f"\nSpeedup ratio (tree_search/model_fit): {avg_ts_time/avg_mf_time:.2f}x")

    # Verify parameter unpacking (modular design)
    if verbose:
        print("\nVerifying parameter unpacking and mapping...")

    # Verify basic arrays match
    assert np.allclose(treesearch.initial_params, defaultmodel.x0), "Initial params don't match!"
    assert np.allclose(treesearch.upper_bound, defaultmodel.ub), "Upper bounds don't match!"
    assert np.allclose(treesearch.lower_bound, defaultmodel.lb), "Lower bounds don't match!"

    # Verify that TreeSearch unpacks parameters consistently
    params = treesearch.initial_params
    control_names = ["pruning_threshold", "stopping_prob", "feature_drop", "lapse_rate", "opp_scale", "center_weight"]
    control_params = {name: float(params[treesearch.param_names.index(name)]) for name in control_names}
    feature_weights = {name: float(params[treesearch.param_names.index(name)]) for name in treesearch.features.keys()}

    if verbose:
        print("  Control parameters:")
        for k, v in control_params.items():
            print(f"    {k:25s} = {v:7.3f}")
        print("  Feature weights:")
        for k, v in feature_weights.items():
            print(f"    {k:25s} = {v:7.3f}")

    # Verify parameter ordering
    expected_order = control_names + sorted(treesearch.features.keys())
    if treesearch.param_names != expected_order:
        print("⚠️ Parameter ordering mismatch!")
        print("  TreeSearch param_names:", treesearch.param_names)
        print("  Expected order:", expected_order)
    else:
        print("✅ Parameter ordering verified: control + feature groups consistent.")
    
    return {
        'match': all_match,
        'nll_tree_search': nll_tree_search[:n_iterations_actual],
        'nll_model_fit': nll_model_fit[:n_iterations_actual],
        'differences': differences,
        'expected_counts_match': expected_counts_match,
        'expected_counts_ts': expected_counts_ts,
        'expected_counts_mf': required_counts_mf,
        'tree_search_times': tree_search_times[:n_iterations_actual],
        'model_fit_times': model_fit_times[:n_iterations_actual],
        'avg_tree_search_time': np.mean(tree_search_times[:n_iterations_actual]) if tree_search_times else 0,
        'avg_model_fit_time': np.mean(model_fit_times[:n_iterations_actual]) if model_fit_times else 0
    }


if __name__ == "__main__":
    # Run verification
    data_folder = "../../monkey_4iar/analysis/data/processed/harry/splits_20000"
    result = verify_implementations(
        data_folder=data_folder,
        fold_idx=0,
        n_trials=5,
        cutoff=1.2,
        manual_seed=1,
        n_iterations=10,
        verbose=True,
        feature_drop=0.0  # Set to 0 to disable feature dropping
    )
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result['match'] and result['expected_counts_match'] else 1)

