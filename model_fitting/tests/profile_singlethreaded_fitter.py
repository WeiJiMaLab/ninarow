"""
Profile SingleThreadedFitter.optimize() / evaluate() with cProfile.

Run from repo root or tests/:
  python profile_singlethreaded_fitter.py
  python profile_singlethreaded_fitter.py --n-iterations 5 --n-trials 20
"""
import cProfile
import io
import pstats
import argparse
import random
import sys
from pathlib import Path

import numpy as np

_MODEL_FITTING = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_FITTING))

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ts = _load("tree_search", _MODEL_FITTING / "tree_search.py")
_tsf = _load("tree_search_fitter", _MODEL_FITTING / "tree_search_fitter.py")
TreeSearch = _ts.TreeSearch
SingleThreadedFitter = _tsf.SingleThreadedFitter

import glob
import pandas as pd

TEMPLATES = {
    "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
    "4IAR": [[1, 1, 1, 1]],
}
WEIGHTS = {"2IAR_CON": 1.0, "2IAR_DIS": 0.4, "3IAR": 3.5, "4IAR": 9.0}


def load_data(data_folder, n_trials=20):
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    if not split_files:
        raise ValueError(f"No split files in {data_folder}")
    df = pd.read_csv(split_files[0])[:n_trials]
    df["expected_counts"] = 1
    if "trial_id" not in df.columns:
        df["trial_id"] = range(len(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-folder", type=str, default=str(_TESTS_DIR / "data"))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-iterations", type=int, default=8)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=35)
    args = ap.parse_args()

    data = load_data(args.data_folder, n_trials=args.n_trials)
    model = TreeSearch(templates=TEMPLATES, initial_weights=WEIGHTS, verbose=False)
    if "feature_drop" in model.param_names:
        idx = model.param_names.index("feature_drop")
        model.initial_params[idx] = 0.0
        model.lower_bound[idx] = 0.0
        model.upper_bound[idx] = 0.0
    fitter = SingleThreadedFitter(model, n_repeats=args.n_repeats, verbose=False)
    fitter.data = data
    params = model.initial_params.copy()

    random.seed(args.seed)
    _ = random.randint(0, 2**64)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.n_iterations):
        fitter.optimize(params)
    pr.disable()

    print("=" * 88)
    print("SingleThreadedFitter — cProfile (tree_search.evaluate → process_single_trial)")
    print("=" * 88)
    print(f"  n_trials={args.n_trials}, n_iterations={args.n_iterations}, "
          f"n_repeats={args.n_repeats}, seed={args.seed}")
    print()

    s1 = io.StringIO()
    pstats.Stats(pr, stream=s1).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top)
    print(f"--- Top {args.top} by CUMULATIVE time (includes callees) ---")
    print(s1.getvalue())

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats(pstats.SortKey.TIME).print_stats(args.top)
    print(f"--- Top {args.top} by SELF time (only in that function) ---")
    print(s2.getvalue())

    print("--- tree_search.py hot spots (filter) ---")
    s3 = io.StringIO()
    pstats.Stats(pr, stream=s3).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(80)
    for line in s3.getvalue().splitlines():
        if "tree_search.py" in line and any(
            x in line for x in ("optimize", "evaluate", "process_single", "predict",
                                  "set_params", "create_heuristic", "get_random", "record")
        ):
            print(line)

    print()
    print("=" * 88)
    print("TAGGED SUMMARY (typical breakdown)")
    print("=" * 88)
    print("""
  [HOT] C++ search     — _swig_fourbynine.AbstractSearch_complete_search
                         + NInARowBestFirstSearch construction (new_*)
  [HOT] Python glue    — tree_search.TreeSearch.predict → complete_search / get_best_move
  [MID] Heuristic build— create_heuristic + feature_generator.create_feature (per set_params)
  [MID] Pandas         — iloc / row access while building shuffled_trials
  [LOW] IBS            — IBSTracker.record (tiny vs predict loop count)
  [LOW] Order          — get_random_order

  Most wall time is in native complete_search; second is allocating BestFirstSearch per predict.
  Run with higher n_repeats to shift time toward complete_search vs fixed costs.
""")


if __name__ == "__main__":
    main()
