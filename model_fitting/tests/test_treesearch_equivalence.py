"""
Verify that SingleThreadedFitter (tree_search.py) and ModelFitter (model_fit.py)
produce identical NLL values.

This test compares the two implementations across multiple optimization iterations
to ensure they produce the same results.

Imports: tree_search and model_fit are loaded only from the sibling files
``model_fitting/tree_search.py`` and ``model_fitting/model_fit.py`` (not from any
other package on sys.path). Shared helpers (parsers, fourbynine) come from
``model_fitting/`` via sys.path.
"""
import importlib.util
import sys
from pathlib import Path
import random
import time
import glob

import numpy as np
import pandas as pd
from prodict import Prodict

_MODEL_FITTING = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent


def _load_module_from_path(unique_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(_MODEL_FITTING))

# Use the real module names ``tree_search`` and ``model_fit`` so that
# ``model_fit.initialize_thread`` (multiprocessing pool initializer) pickles;
# worker processes must be able to ``import model_fit``.
_ts = _load_module_from_path("tree_search", _MODEL_FITTING / "tree_search.py")
_mf = _load_module_from_path("model_fit", _MODEL_FITTING / "model_fit.py")

TreeSearch = _ts.TreeSearch
SingleThreadedFitter = _ts.SingleThreadedFitter
DefaultModel = _mf.DefaultModel
ModelFitter = _mf.ModelFitter
SuccessFrequencyTracker = _mf.SuccessFrequencyTracker

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


def run_tree_search_optimization(fitter, params, n_iterations, manual_seed, track_seeds=False):
    """Run optimization iterations with SingleThreadedFitter."""
    random.seed(manual_seed)
    _ = random.randint(0, 2**64)  # Match worker state

    seeds_generated = []
    nlls, times = [], []
    for i in range(n_iterations):
        if track_seeds:
            state_before = random.getstate()

        start = time.perf_counter()
        nll = fitter.optimize(params)
        times.append(time.perf_counter() - start)
        nlls.append(nll)

        if track_seeds:
            state_after = random.getstate()
            seeds_generated.append((i, state_before, state_after))

    if track_seeds:
        return nlls, times, seeds_generated
    return nlls, times


def run_model_fit_optimization(fitter, move_tasks, params, n_iterations, manual_seed):
    """Run optimization iterations with model_fit.py (ModelFitter)."""
    random.seed(manual_seed)
    _mf.initialize_thread_pool(1, manual_seed=manual_seed)

    nlls, times = [], []
    for _ in range(n_iterations):
        start = time.perf_counter()
        loglik = sum(fitter.compute_loglik(move_tasks, params).values())
        times.append(time.perf_counter() - start)
        nlls.append(loglik)
    return nlls, times


def compare_results(nll_st, nll_mf, tolerance=1e-5, verbose=True):
    """Compare NLL values from tree_search.SingleThreadedFitter and model_fit.ModelFitter."""
    matches = []

    if verbose:
        print("\n" + "=" * 90)
        print(f"{'Iter':>4} | {'tree_search (ST)':>18} | {'model_fit':>15} | {'Diff':>12} |")
        print("-" * 90)

    for i, (st, mf) in enumerate(zip(nll_st, nll_mf)):
        diff = abs(st - mf)
        matches.append(diff <= tolerance)

        if verbose:
            status = "✓" if diff <= tolerance else "✗"
            print(f"{i:>4} | {st:>18.6f} | {mf:>15.6f} | {diff:>12.2e} {status}")

    all_match = all(matches)

    if verbose:
        print("\nSUMMARY:")
        status = "✅" if all_match else "❌"
        print(f"{status}  tree_search vs model_fit: {sum(matches)}/{len(matches)} match")

    return all_match


def verify_equivalence(data_folder, fold_idx=0, n_trials=20, cutoff=100.0,
                       manual_seed=1, n_iterations=20, verbose=True, feature_drop=0.0):
    """Verify equivalence between SingleThreadedFitter (tree_search) and ModelFitter (model_fit)."""

    if verbose:
        print("=" * 80)
        print("VERIFICATION: tree_search.SingleThreadedFitter vs model_fit.ModelFitter")
        print("=" * 80)
        print(f"Data folder: {data_folder}")
        print(f"Trials: {n_trials}, Iterations: {n_iterations}, Seed: {manual_seed}")
        print()

    data = load_data(data_folder, fold_idx, n_trials)
    data["expected_counts"] = 1
    if 'trial_id' not in data.columns:
        data['trial_id'] = range(len(data))

    templates = {
        "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        "4IAR": [[1, 1, 1, 1]],
    }
    weights = {"2IAR_CON": 1.0, "2IAR_DIS": 0.4, "3IAR": 3.5, "4IAR": 9.0}

    model_st = TreeSearch(templates=templates, initial_weights=weights)
    model_st.cutoff = cutoff
    if 'feature_drop' in model_st.param_names:
        idx = model_st.param_names.index('feature_drop')
        model_st.initial_params[idx] = feature_drop
        model_st.lower_bound[idx] = feature_drop
        model_st.upper_bound[idx] = feature_drop
    fitter_st = SingleThreadedFitter(model_st, n_repeats=1, verbose=False)
    fitter_st.data = data.copy()

    model_mf = DefaultModel()
    model_mf.cutoff = cutoff
    model_mf.x0 = model_st.initial_params.copy()
    model_mf.ub = model_st.upper_bound.copy()
    model_mf.lb = model_st.lower_bound.copy()
    model_mf.pub = model_st.plausible_upper_bound.copy()
    model_mf.plb = model_st.plausible_lower_bound.copy()

    fitter_mf = ModelFitter(
        args=Prodict({'threads': 1, 'random_sample': False, 'verbose': False}),
        model=model_mf
    )
    moves = parse_dataframe_to_moves(data)
    move_tasks = {
        move: SuccessFrequencyTracker(model_mf.expt_factor)
        for move in moves
    }
    for task in move_tasks.values():
        task.required_success_count = 1

    test_params = model_st.initial_params.copy()

    if verbose:
        print("Running SingleThreadedFitter (tree_search.py)...")
    nll_st, times_st = run_tree_search_optimization(fitter_st, test_params, n_iterations, manual_seed)

    if verbose:
        print("Running ModelFitter (model_fit.py)...")
    nll_mf, times_mf = run_model_fit_optimization(fitter_mf, move_tasks, test_params, n_iterations, manual_seed)

    all_match = compare_results(nll_st, nll_mf, verbose=verbose)

    if verbose:
        print("\n" + "=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        for name, times in [("tree_search (SingleThreadedFitter)", times_st), ("model_fit (ModelFitter)", times_mf)]:
            avg = np.mean(times)
            total = np.sum(times)
            print(f"{name:<40} {avg*1000:>15.2f} ms/iter {total:>15.4f} s total")

    return all_match


def test_singlethreaded_vs_model_fit_nll_equivalence():
    """Pytest entry: same defaults as ``python test_treesearch_equivalence.py``."""
    assert verify_equivalence(
        data_folder=str(_TESTS_DIR / "data"),
        n_trials=20,
        n_iterations=20,
        manual_seed=1,
        verbose=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Verify NLL equivalence: tree_search.SingleThreadedFitter vs model_fit.ModelFitter'
    )
    parser.add_argument('--data-folder', type=str,
                        default=str(_TESTS_DIR / "data"),
                        help='Path to directory containing split_*.csv files')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials to use for testing')
    parser.add_argument('--n-iterations', type=int, default=20, help='Number of optimization iterations to test')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary of results, suppress iteration details')

    args = parser.parse_args()

    success = verify_equivalence(
        data_folder=args.data_folder,
        n_trials=args.n_trials,
        n_iterations=args.n_iterations,
        manual_seed=args.seed,
        verbose=not args.summary_only
    )

    sys.exit(0 if success else 1)
