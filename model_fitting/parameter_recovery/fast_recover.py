"""Round-trip parameter recovery: N synthetic "participants" built from real
board positions in data/sample, each with its own Sobol-drawn ground-truth
theta, refit blind.

Uses the DEFAULT (OG 4-group) TreeSearch model and its config.yaml bounds
directly, rather than the 6-template recovery model in recovery_common.py --
this is a fast sanity check of the pipeline, not the production-scale
validation.

Ground truth is a fixed Sobol design over the model's plausible box, generated
once (see ground_truth()) so every participant/index refers to the same draw
regardless of how many run or in what order. Board positions are resampled
(with replacement across participants, without replacement within one) from
the two real data/sample participants pooled together -- with N > 2 synthetic
participants we necessarily reuse real boards across several synthetic ones,
which is fine: recovery only needs realistic board *positions*, not distinct
real participants per synthetic one.

Output (mirrors data/sample's layout, one directory per synthetic participant):
    data/recovery/participant<i>/0.csv           synthetic trials (same schema
                                                  as data/sample: black/white/
                                                  move/color/trial_id/n_pieces)
    data/recovery/participant<i>/recovery.json   theta_true/theta_hat/metadata

Run everything sequentially (small N, interactive):
    python fast_recover.py --n-participants 4 --out-dir ../../data/recovery

Run one participant (SLURM array task -- see submit_fast_recovery_array.sh):
    python fast_recover.py --n-participants 30 --index "$SLURM_ARRAY_TASK_ID" \\
        --out-dir ../../data/recovery
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

# Make the sibling ninarow model_fitting importable regardless of CWD.
_MODEL_FITTING = Path(__file__).resolve().parent.parent
if str(_MODEL_FITTING) not in sys.path:
    sys.path.insert(0, str(_MODEL_FITTING))

from tree_search import TreeSearch  # noqa: E402
from tree_search_fitter import MultiThreadedFitter  # noqa: E402
from fourbynine import fourbynine_board, fourbynine_pattern  # noqa: E402

_REPO_ROOT = _MODEL_FITTING.parent
_SAMPLE_DIR = _REPO_ROOT / "data" / "sample"

with open(_MODEL_FITTING / "config.yaml") as _f:
    _BADS_DEFAULTS = yaml.safe_load(_f)["bads"]


def pooled_sample_boards():
    """All data/sample participants' fold-0 boards, pooled into one table."""
    csvs = sorted(_SAMPLE_DIR.glob("participant*/0.csv"))
    if not csvs:
        raise FileNotFoundError(f"No participant*/0.csv under {_SAMPLE_DIR}")
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def boards_for_participant(pool, n_boards, index, seed):
    """Deterministic per-index board subsample (without replacement within one
    participant; different participants draw independent subsamples, so boards
    repeat across the full N > len(pool) but not within a single fold)."""
    n = min(n_boards, len(pool))
    return pool.sample(n=n, random_state=seed * 10_000 + index).reset_index(drop=True)


def ground_truth(model, n_participants, seed=0):
    """Fixed Sobol design over the model's plausible box -- same draw regardless
    of how many participants actually get run (array tasks stay consistent)."""
    d = len(model.param_names)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    unit = sampler.random(n_participants)
    plb, pub = model.plausible_lower_bound, model.plausible_upper_bound
    return qmc.scale(unit, plb, pub)


def generate_synthetic(model, theta, boards, seed):
    """Sample one synthetic move per board row, keeping the sample-data schema
    (black/white/move/color/trial_id/n_pieces) so recovery output is a drop-in
    mirror of data/sample."""
    model.set_params(np.asarray(theta, dtype=np.float64))
    model.heuristic.seed_generator(int(seed))
    moves = np.empty(len(boards), dtype=np.int64)
    for i, row in enumerate(boards.itertuples()):
        board = fourbynine_board(fourbynine_pattern(int(row.black)), fourbynine_pattern(int(row.white)))
        moves[i] = 1 << int(model.predict(board))
    out = pd.DataFrame({
        "black": boards["black"].astype("int64").values,
        "white": boards["white"].astype("int64").values,
        "move": moves,
        "color": boards["color"].values,
    })
    for passthrough in ("trial_id", "n_pieces"):
        if passthrough in boards.columns:
            out[passthrough] = boards[passthrough].values
    return out


def run_one(index, theta_true, pool, args, names):
    out_participant_dir = Path(args.out_dir) / f"participant{index}"
    out_participant_dir.mkdir(parents=True, exist_ok=True)

    boards = boards_for_participant(pool, args.n_boards, index, args.seed)
    gen_model = TreeSearch(verbose=False)
    synthetic = generate_synthetic(gen_model, theta_true, boards, seed=args.seed * 1000 + index)
    synthetic.to_csv(out_participant_dir / "0.csv", index=False)

    bads_options = dict(_BADS_DEFAULTS)
    bads_options["tol_fun"] = 1e-7
    bads_options["max_fun_evals"] = args.max_evals

    fit_model = TreeSearch(verbose=False)
    fitter = MultiThreadedFitter(fit_model, verbose=False, n_repeats=args.n_repeats, n_workers=args.n_workers)
    t0 = time.time()
    theta_hat, train_nll = fitter.fit(
        synthetic, bads_options=bads_options, atol_mesh=5e-3, atol_fun=1e-7,
    )
    elapsed = time.time() - t0
    try:
        fitter.close()
    except Exception:
        pass

    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    train_nll_total = float(np.sum(train_nll)) if np.ndim(train_nll) else float(train_nll)

    result = {
        "theta_id": f"participant{index}",
        "index": index,
        "param_names": names,
        "theta_true": np.asarray(theta_true, dtype=np.float64).tolist(),
        "theta_hat": theta_hat.tolist(),
        "train_nll_total": train_nll_total,
        "n_trials": int(len(synthetic)),
        "n_repeats": args.n_repeats,
        "max_evals": args.max_evals,
        "elapsed_sec": elapsed,
    }
    out_path = out_participant_dir / "recovery.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[participant{index}] {elapsed:.1f}s  theta_true={np.round(theta_true, 3)}")
    print(f"        theta_hat ={np.round(theta_hat, 3)}  -> {out_participant_dir}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-participants", type=int, default=30,
                        help="Size of the fixed Sobol ground-truth design.")
    parser.add_argument("--index", type=int, default=None,
                        help="Run only this one participant index (SLURM array task). "
                             "Omit to run all --n-participants sequentially.")
    parser.add_argument("--n-boards", type=int, default=30,
                        help="Boards per synthetic participant (resampled from the pooled "
                             "data/sample boards; reused across participants once N exceeds the pool).")
    parser.add_argument("--n-repeats", type=int, default=12, help="Max IBS repeats.")
    parser.add_argument("--max-evals", type=int, default=200, help="BADS max_fun_evals.")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "data" / "recovery"))
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    pool = pooled_sample_boards()
    probe = TreeSearch(verbose=False)
    names = probe.param_names
    thetas = ground_truth(probe, args.n_participants, seed=args.seed)

    indices = [args.index] if args.index is not None else list(range(args.n_participants))
    if args.index is not None and not (0 <= args.index < args.n_participants):
        raise SystemExit(f"--index {args.index} out of range [0, {args.n_participants})")

    t_start = time.time()
    for i in indices:
        run_one(i, thetas[i], pool, args, names)
    print(f"\nTotal: {time.time() - t_start:.1f}s for {len(indices)} participant(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
