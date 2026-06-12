"""Recover one ground-truth parameter vector (one SLURM array task).

Pipeline for a single theta_true:
  1. Load its row from the ground-truth JSONL.
  2. Simulate synthetic choices on the board positions (generate.py).
  3. Refit a fresh 6-template TreeSearch with the production BADS+IBS settings,
     starting from default initial params (NOT theta_true).
  4. Save {theta_true, theta_hat, train NLL, metadata} to a JSON.

Single pooled fit per theta (no 5x3 CV).

Example (one task):
    python recover.py --ground-truth ground_truth.jsonl --jobid 0 \
        --boards /path/to/modeling_2500 --out-dir results

Smoke test (cheap):
    python recover.py --ground-truth ground_truth.jsonl --jobid 0 \
        --boards /path/to/modeling_2500 --out-dir results_smoke \
        --max-trials 60 --max-evals 10 --n-repeats 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_common import (
    DEFAULT_ATOL_FUN,
    DEFAULT_ATOL_MESH,
    DEFAULT_BADS_OPTIONS,
    DEFAULT_N_REPEATS,
    build_model,
    load_boards,
    param_names,
)
from generate import generate_synthetic_dataset

from tree_search_fitter import MultiThreadedFitter


def load_ground_truth(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--jobid", type=int, default=None,
                        help="Index into the ground-truth file (0-based). "
                             "Defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--theta-id", type=str, default=None,
                        help="Select by theta_id instead of --jobid.")
    parser.add_argument("--boards", required=True,
                        help="CSV file or directory of board positions (e.g. modeling_2500).")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Subsample boards to this many positions (cost control).")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Pool size; defaults to $SLURM_CPUS_PER_TASK else 6.")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override BADS max_fun_evals (use small values for smoke tests).")
    parser.add_argument("--gen-seed", type=int, default=12345,
                        help="Base seed for synthetic-data generation (offset by theta index).")
    args = parser.parse_args()

    import os
    if args.n_workers is not None:
        n_workers = args.n_workers
    elif "SLURM_CPUS_PER_TASK" in os.environ:
        n_workers = int(os.environ["SLURM_CPUS_PER_TASK"].split(",")[0])
    else:
        n_workers = 6

    ground_truth = load_ground_truth(args.ground_truth)
    if args.theta_id is not None:
        matches = [(i, r) for i, r in enumerate(ground_truth) if r["theta_id"] == args.theta_id]
        if not matches:
            raise SystemExit(f"theta_id {args.theta_id!r} not found")
        index, record = matches[0]
    else:
        index = args.jobid
        if index is None:
            index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if index >= len(ground_truth):
            raise SystemExit(f"jobid {index} out of range ({len(ground_truth)} points)")
        record = ground_truth[index]

    theta_id = record["theta_id"]
    names = param_names()
    if record["param_names"] != names:
        raise SystemExit(
            f"Ground-truth param order mismatch.\n  file: {record['param_names']}\n  model: {names}"
        )
    theta_true = np.asarray(record["params"], dtype=np.float64)

    print("=" * 80)
    print(f"  Parameter recovery: theta_id={theta_id} (index {index})")
    print(f"  n_workers={n_workers}  n_repeats={args.n_repeats}  max_trials={args.max_trials}")
    print("=" * 80)

    # 1. Boards (deterministic subsample if capped).
    boards = load_boards(args.boards, max_trials=args.max_trials, seed=index)
    print(f"  boards: {len(boards)} positions from {args.boards}")

    # 2. Simulate synthetic choices at theta_true.
    gen_seed = args.gen_seed + index
    synthetic = generate_synthetic_dataset(theta_true, boards, seed=gen_seed)
    print(f"  generated {len(synthetic)} synthetic moves (gen_seed={gen_seed})")

    # 3. Refit a fresh model from default initial params.
    bads_options = dict(DEFAULT_BADS_OPTIONS)
    if args.max_evals is not None:
        bads_options["max_fun_evals"] = args.max_evals

    model = build_model(verbose=False)
    fitter = MultiThreadedFitter(model, verbose=True, n_repeats=args.n_repeats, n_workers=n_workers)

    t0 = time.time()
    theta_hat, train_nll = fitter.fit(
        synthetic,
        bads_options=bads_options,
        atol_mesh=DEFAULT_ATOL_MESH,
        atol_fun=DEFAULT_ATOL_FUN,
    )
    elapsed = time.time() - t0
    try:
        fitter.close()
    except Exception:
        pass

    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    train_nll_total = float(np.sum(train_nll)) if np.ndim(train_nll) else float(train_nll)

    # 4. Persist.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "theta_id": theta_id,
        "index": index,
        "param_names": names,
        "theta_true": theta_true.tolist(),
        "theta_hat": theta_hat.tolist(),
        "train_nll_total": train_nll_total,
        "n_trials": int(len(synthetic)),
        "n_repeats": args.n_repeats,
        "gen_seed": gen_seed,
        "max_evals": args.max_evals,
        "elapsed_sec": elapsed,
        "boards": str(args.boards),
    }
    out_path = out_dir / f"recovery_{theta_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  theta_true: {np.round(theta_true, 3)}")
    print(f"  theta_hat : {np.round(theta_hat, 3)}")
    print(f"  elapsed   : {elapsed/60:.1f} min  ->  {out_path}")


if __name__ == "__main__":
    main()
