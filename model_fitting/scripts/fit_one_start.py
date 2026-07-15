"""Fit ONE multistart start for a held-out CV fold; writes starts/<held_out>.<start>.json.

Invoked once per SLURM array task by fit_all.py's parallel path (task id ->
start index). Not meant to be run standalone outside that array, though it works fine
run directly for debugging (pass --start explicitly).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multistart import default_model_factory, fit_one_start, write_start_json

REQUIRED_COLUMNS = {"black", "white", "move", "color"}


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
    parser.add_argument("data_dir", type=str)
    parser.add_argument("n_splits", type=int)
    parser.add_argument("held_out_index", type=int)
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start index. Defaults to SLURM_ARRAY_TASK_ID when omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Pool size (default: SLURM_CPUS_PER_TASK if set, else 6).")
    parser.add_argument("--n-repeats", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = args.start
    if start is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            sys.exit("--start not given and SLURM_ARRAY_TASK_ID is not set.")
        start = int(os.environ["SLURM_ARRAY_TASK_ID"])

    data_dir = Path(args.data_dir)
    split_paths = check_data_dir(data_dir, args.n_splits)
    train, _test = load_split(split_paths, args.held_out_index)

    model = default_model_factory(verbose=args.verbose)()
    print(f"fit_one_start: held_out={args.held_out_index} start={start} train_rows={len(train)}")
    record = fit_one_start(
        model, train, start, seed=args.seed,
        n_workers=args.n_workers, n_repeats=args.n_repeats, verbose=args.verbose,
    )

    starts_dir = data_dir / "starts"
    starts_dir.mkdir(exist_ok=True)
    out_path = starts_dir / f"{args.held_out_index}.{start}.json"
    write_start_json(out_path, record)
    print(f"fit_one_start: wrote {out_path} (train_nll={record['train_nll_raw']:.4f})")


if __name__ == "__main__":
    main()
