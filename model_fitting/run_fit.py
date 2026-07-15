"""Cross-validation fitting entry point for a pre-split fold directory.

Usage:
    python run_fit.py <data_dir> <n_splits> <held_out_index>

<data_dir> must contain <n_splits> files named 0.csv, 1.csv, ..., (n_splits-1).csv
(the same per-move split format produced by monkey_4iar's fold generation, i.e.
columns black/white/move/color[, trial_id, n_pieces]). The fold at <held_out_index>
is held out as the test set; the rest are concatenated as the training set.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tree_search import TreeSearch
from tree_search_fitter import MultiThreadedFitter

REQUIRED_COLUMNS = {"black", "white", "move", "color"}


def default_n_workers():
    """SLURM_CPUS_PER_TASK if running inside a job allocation, else a small fixed
    default. MultiThreadedFitter's own default (n_workers<=0 -> os.cpu_count()) is NOT
    safe to use directly outside a job: on a shared login/compute node os.cpu_count()
    reports the WHOLE machine's core count, not any per-job allocation."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"].split(",")[0])
    return 6


def check_data_dir(data_dir, n_splits):
    """Verify data_dir exists and contains exactly the expected 0..n_splits-1 fold files."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    expected = [data_dir / f"{i}.csv" for i in range(n_splits)]
    missing = [p for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected {n_splits} split file(s) in {data_dir}, missing: "
            + ", ".join(p.name for p in missing)
        )

    found = sorted(p.name for p in data_dir.glob("*.csv"))
    expected_names = sorted(p.name for p in expected)
    if found != expected_names:
        raise ValueError(
            f"{data_dir} does not contain exactly the expected {n_splits} split(s). "
            f"Expected {expected_names}, found {found}."
        )

    print(f"Found expected {n_splits} split(s) in {data_dir}")
    return expected


def read_fold_csv(path):
    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing columns: {missing_columns}")
    return df


def load_split(split_paths, held_out_index):
    """(train, test): test = the held-out fold; train = the rest, concatenated."""
    folds = [read_fold_csv(p) for p in split_paths]
    test = folds[held_out_index]
    train = pd.concat(
        [fold for i, fold in enumerate(folds) if i != held_out_index]
    ).reset_index(drop=True)
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=str, help="Directory containing 0.csv..N-1.csv split files.")
    parser.add_argument("n_splits", type=int, help="Expected number of splits.")
    parser.add_argument("held_out_index", type=int, help="Index of the split to hold out as the test set.")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Pool size for MultiThreadedFitter (default: SLURM_CPUS_PER_TASK if set, else 6).")
    parser.add_argument("--n-repeats", type=int, default=50, help="Max IBS repeats at the fine BADS polls.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (0 <= args.held_out_index < args.n_splits):
        sys.exit(f"held_out_index must be in [0, {args.n_splits}), got {args.held_out_index}")

    split_paths = check_data_dir(data_dir, args.n_splits)
    train, test = load_split(split_paths, args.held_out_index)
    print(f"Held-out split: {args.held_out_index} ({len(test)} rows); train: {len(train)} rows")

    n_workers = args.n_workers if args.n_workers is not None else default_n_workers()
    print(f"Using n_workers={n_workers}")
    model = TreeSearch(verbose=args.verbose)
    fitter = MultiThreadedFitter(model, verbose=args.verbose, n_repeats=args.n_repeats, n_workers=n_workers)
    bads_options = {"uncertainty_handling": True, "specify_target_noise": True, "display": "iter"}
    try:
        params, train_ll = fitter.fit(train, bads_options=bads_options)
        print("Finished fitting, evaluating held-out split...")
        test_ll, _ = fitter.evaluate(params, test)
    finally:
        fitter.close()

    print(f"Fitted params: {params}")
    print(f"Train NLL: {sum(train_ll):.4f}")
    print(f"Test NLL: {sum(test_ll):.4f}")
    return params, train_ll, test_ll


if __name__ == "__main__":
    main()
