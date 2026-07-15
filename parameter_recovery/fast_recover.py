"""Round-trip parameter recovery: N synthetic "participants" built from real
board positions in data/sample, each with its own Sobol-drawn ground-truth
theta, refit blind through the SAME multistart hot path fit_all.py uses
(model_fitting/multistart.py: fit_one_start per start, then select_winner's
high-repeat re-evaluation to pick the argmin).

This checks a recoverability LOWER BOUND, not full recovery: --n-starts,
--n-repeats, and --n-boards are all smaller than production (fit_all.py
defaults: 5 starts, 40 repeats, full participant datasets), and --max-evals
caps BADS well short of its own tol_mesh/tol_fun convergence (production
leaves this uncapped). If recovery looks reasonable even under-resourced like
this, that's a good sign; a bad result here doesn't prove production fails,
since production gets more starts/repeats/data/eval budget to work with.
Pass --max-evals=-1 (and turn up the others) to remove the cap and match
production exactly, at production's cost (~1 day/start uncapped).

--exclude-feature-drop mirrors monkey_4iar's frozen-feature_drop regime:
drops feature_drop from the fitted vector entirely (fixed at 0.0) rather than
merely pinning it to config.yaml's narrow plausible range while BADS still
searches it as a free dimension. Threaded through generation AND fitting (both
must agree on the parameter space), so ground truth is drawn in the same
reduced space the fit uses.

Uses the DEFAULT (OG 4-group) TreeSearch model and its config.yaml bounds
directly, rather than the 6-template recovery model in recovery_common.py --
this is a fast sanity check of the pipeline, not the production-scale
validation.

Output (mirrors data/sample's layout, one directory per synthetic participant):
    data/recovery/participant<i>/0.csv           synthetic trials (same schema
                                                  as data/sample: black/white/
                                                  move/color/trial_id/n_pieces)
    data/recovery/participant<i>/recovery.json   theta_true/theta_hat/metadata
                                                  (+ n_starts/winning_start, so
                                                  it's visible this went through
                                                  the same multistart+select
                                                  path as a real fit_all run)

Run everything sequentially (small N, interactive):
    python fast_recover.py --n-participants 4 --out-dir ../data/recovery

Run one participant (SLURM array task -- see submit_recovery.sh):
    python fast_recover.py --n-participants 30 --index "$SLURM_ARRAY_TASK_ID" \\
        --out-dir ../data/recovery
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the sibling ninarow model_fitting importable regardless of CWD.
_MODEL_FITTING = _REPO_ROOT / "model_fitting"
if str(_MODEL_FITTING) not in sys.path:
    sys.path.insert(0, str(_MODEL_FITTING))

from tree_search import TreeSearch  # noqa: E402
from fourbynine import fourbynine_board, fourbynine_pattern  # noqa: E402
from multistart import (  # noqa: E402
    default_model_factory,
    default_n_workers,
    fit_one_start,
    select_winner,
    to_array,
    write_start_json,
)

_SAMPLE_DIR = _REPO_ROOT / "data" / "sample"


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
    """One synthetic participant, fit through the exact fit_all.py hot path:
    n_starts independent fit_one_start calls (start 0 = fixed initial_params,
    starts >=1 = uniform-in-plausible-box), then select_winner re-evaluates
    every start's params at high IBS repeats and picks the argmin -- the same
    two-stage procedure a real (participant, held-out fold) grid point uses,
    just with smaller n_starts/n_repeats/max_evals for speed."""
    out_participant_dir = Path(args.out_dir) / f"participant{index}"
    out_participant_dir.mkdir(parents=True, exist_ok=True)

    boards = boards_for_participant(pool, args.n_boards, index, args.seed)
    gen_model = TreeSearch(verbose=False, exclude_feature_drop=args.exclude_feature_drop)
    synthetic = generate_synthetic(gen_model, theta_true, boards, seed=args.seed * 1000 + index)
    synthetic.to_csv(out_participant_dir / "0.csv", index=False)

    n_workers = args.n_workers if args.n_workers is not None else default_n_workers()
    bads_overrides = {"max_fun_evals": args.max_evals} if args.max_evals and args.max_evals > 0 else None
    model_factory = default_model_factory(verbose=False, exclude_feature_drop=args.exclude_feature_drop)

    t0 = time.time()
    starts = []
    for start in range(args.n_starts):
        model = model_factory()
        record = fit_one_start(
            model, synthetic, start, seed=args.seed * 1000 + index,
            n_workers=n_workers, n_repeats=args.n_repeats, verbose=False,
            bads_options_overrides=bads_overrides,
        )
        write_start_json(out_participant_dir / f"start_{start}.json", record)
        starts.append(record)

    winner_idx, winner, train_nll, _test_nll = select_winner(
        model_factory, starts, synthetic, synthetic,
        n_workers=n_workers, verbose=False,
    )
    elapsed = time.time() - t0

    theta_hat = to_array(winner["params"], names)
    train_nll_total = float(np.sum(train_nll))

    result = {
        "theta_id": f"participant{index}",
        "index": index,
        "param_names": names,
        "theta_true": np.asarray(theta_true, dtype=np.float64).tolist(),
        "theta_hat": theta_hat.tolist(),
        "train_nll_total": train_nll_total,
        "n_trials": int(len(synthetic)),
        "n_starts": args.n_starts,
        "winning_start": winner["start"],
        "winner_rank": winner_idx,
        "n_repeats": args.n_repeats,
        "max_evals": args.max_evals if bads_overrides else None,
        "exclude_feature_drop": args.exclude_feature_drop,
        "elapsed_sec": elapsed,
    }
    out_path = out_participant_dir / "recovery.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[participant{index}] {elapsed:.1f}s  {args.n_starts} starts, winner=start {winner['start']}")
    print(f"        theta_true={np.round(theta_true, 3)}")
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
    parser.add_argument("--n-starts", type=int, default=3,
                        help="Multistarts per participant (fit_all.py production default: 5).")
    parser.add_argument("--n-repeats", type=int, default=12,
                        help="Max IBS repeats during the fit (fit_all.py production default: 40).")
    parser.add_argument("--max-evals", type=int, default=300,
                        help="BADS max_fun_evals cap (production leaves this unset -- BADS runs "
                             "to tol_mesh/tol_fun convergence, which is far slower). Capping trades "
                             "convergence for speed: this checks a recoverability LOWER BOUND (can "
                             "the procedure get in the right neighborhood fast?), not full recovery. "
                             "Pass --max-evals=-1 to disable the cap and match production exactly.")
    parser.add_argument("--exclude-feature-drop", action="store_true",
                        help="Mirror monkey_4iar's frozen-feature_drop regime: drop it from the "
                             "fitted vector entirely (fixed at 0.0) instead of merely pinning it to "
                             "a narrow plausible range while BADS still searches it as a free dim.")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Defaults to SLURM_CPUS_PER_TASK if set, else 6 (multistart.default_n_workers).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "data" / "recovery"))
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    pool = pooled_sample_boards()
    probe = TreeSearch(verbose=False, exclude_feature_drop=args.exclude_feature_drop)
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
