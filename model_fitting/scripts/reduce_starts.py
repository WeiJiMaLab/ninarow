"""Consolidate a fold's starts/<held_out>.<start>.json files into a winner.

Used by run_multistart.py after all starts finish (sequential or parallel/sbatch).
Can also be run standalone once a starts/ directory is fully populated.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multistart import default_model_factory, select_winner, validate_starts, write_result_json
from run_fit import check_data_dir, load_split


def reduce_starts(data_dir, n_splits, held_out_index, n_starts, n_workers=None,
                   reeval_repeats=None, n_evals=None, allow_partial=False, verbose=False):
    data_dir = Path(data_dir)
    split_paths = check_data_dir(data_dir, n_splits)
    train, test = load_split(split_paths, held_out_index)

    starts_dir = data_dir / "starts"
    files = sorted(glob.glob(str(starts_dir / f"{held_out_index}.*.json")))
    if not files:
        raise SystemExit(f"reduce: no starts found for held_out={held_out_index} in {starts_dir}")
    starts = [json.load(open(fp)) for fp in files]
    starts = validate_starts(starts, n_starts, f"held_out={held_out_index}", allow_partial=allow_partial)

    kwargs = {}
    if reeval_repeats is not None:
        kwargs["reeval_repeats"] = reeval_repeats
    if n_evals is not None:
        kwargs["n_evals"] = n_evals

    winner_idx, winner, train_nll, test_nll = select_winner(
        default_model_factory(verbose=verbose), starts, train, test,
        n_workers=n_workers, verbose=verbose, **kwargs,
    )
    result_path = data_dir / "results" / f"{held_out_index}.json"
    write_result_json(result_path, held_out_index, winner, train_nll, test_nll, n_starts)
    print(f"reduce: winner=start {winner['start']} ({winner_idx + 1}/{len(starts)} clean starts)")
    print(f"Fitted params: {winner['params']}")
    print(f"Train NLL: {sum(train_nll):.4f}")
    print(f"Test NLL: {sum(test_nll):.4f}")
    print(f"Results written to {result_path}")
    return winner, train_nll, test_nll


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("n_splits", type=int)
    parser.add_argument("held_out_index", type=int)
    parser.add_argument("n_starts", type=int)
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Pool size (default: SLURM_CPUS_PER_TASK if set, else 6).")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    reduce_starts(
        args.data_dir, args.n_splits, args.held_out_index, args.n_starts,
        n_workers=args.n_workers, allow_partial=args.allow_partial, verbose=args.verbose,
    )
