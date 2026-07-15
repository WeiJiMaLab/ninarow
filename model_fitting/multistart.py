"""Multistart fitting primitives shared by scripts/run_multistart.py (interactive
driver) and scripts/fit_one_start.py (per-start worker invoked by sbatch).

Mirrors monkey_4iar's src/models.py multistart machinery (start_x0 / write_start_json /
validate_starts / re-eval + argmin winner selection), reimplemented here so ninarow
stays self-contained (no reverse dependency on monkey_4iar).
"""

import json
import os
from pathlib import Path

import numpy as np

from tree_search import TreeSearch
from tree_search_fitter import MultiThreadedFitter
from run_fit import default_n_workers

# Re-evaluation config for winner selection: a single fit's IBS NLL is too noisy to
# argmin directly (the start that got lucky noise looks best, not the one that's
# actually best), so every start is re-scored at higher IBS repeats before picking.
REEVAL_REPEATS = 50
REEVAL_N_EVALS = 5


def start_x0(start, model, seed=0):
    """x0 for a start: start 0 = the model's fixed initial_params; start s>=1 = uniform
    in the model's plausible box, seeded deterministically from (seed, start)."""
    if start == 0:
        return np.asarray(model.initial_params, dtype=float).copy()
    rng = np.random.default_rng(seed + start)
    plb = np.asarray(model.plausible_lower_bound)
    pub = np.asarray(model.plausible_upper_bound)
    return rng.uniform(plb, pub).astype(np.float32)


def _atomic_write_json(path, record):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_start_json(path, record):
    """Persist one multistart's result atomically with a success marker, so a start
    file's mere existence means "this start finished cleanly" (tmp -> fsync -> rename)."""
    _atomic_write_json(path, {**record, "status": "ok"})


def write_result_json(path, held_out_index, winner, train_nll, test_nll, n_starts):
    """Persist the winning fit for one (participant, held-out fold) grid point.

    train_nll is collapsed to a scalar (sum over training trials); test_nll is kept
    as the full per-trial array (needed for downstream bootstrapping over held-out
    trials, not just its sum).
    """
    record = {
        "held_out_index": held_out_index,
        "winning_start": winner["start"],
        "n_starts": n_starts,
        "params": winner["params"],
        "x0": winner["x0"],
        "train_nll_raw": winner["train_nll_raw"],
        "train_nll_reeval": winner.get("train_nll_reeval"),
        "train_nll": float(np.sum(train_nll)),
        "test_nll_per_trial": np.asarray(test_nll, dtype=float).tolist(),
        "test_nll": float(np.sum(test_nll)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, record)
    return record


def _start_is_clean(record):
    if record.get("status", "ok") != "ok":
        return False
    params = np.asarray(record.get("params", []), dtype=float)
    return (
        params.size > 0
        and bool(np.all(np.isfinite(params)))
        and np.isfinite(record.get("train_nll_raw", np.nan))
    )


def validate_starts(starts, expected, label, allow_partial=False):
    """Keep only clean starts; refuse (or warn, if allow_partial) when fewer than
    `expected` finished cleanly."""
    clean = [s for s in starts if _start_is_clean(s)]
    if len(clean) < expected:
        msg = (
            f"{label}: only {len(clean)}/{expected} multistarts finished cleanly "
            f"({len(starts)} file(s), {len(starts) - len(clean)} invalid/failed)"
        )
        if not allow_partial:
            raise SystemExit(msg + " — refusing to reduce a partial set.")
        print(f"WARNING {msg} — proceeding on the clean subset.", flush=True)
    return clean


def fit_one_start(model, train_data, start, seed=0, n_workers=None, n_repeats=50, verbose=False):
    """Run a single BADS start from start_x0(start, model, seed). Returns a start record
    (x0, params, train_nll_raw) — the same schema write_start_json persists."""
    if n_workers is None:
        n_workers = default_n_workers()
    x0 = start_x0(start, model, seed)
    model.initial_params = x0
    fitter = MultiThreadedFitter(model, verbose=verbose, n_repeats=n_repeats, n_workers=n_workers)
    bads_options = {"uncertainty_handling": True, "specify_target_noise": True, "display": "iter" if verbose else "off"}
    try:
        params, train_ll = fitter.fit(train_data, bads_options=bads_options)
    finally:
        fitter.close()
    return {
        "start": start,
        "x0": np.asarray(x0, dtype=float).tolist(),
        "params": np.asarray(params, dtype=float).tolist(),
        "train_nll_raw": float(np.sum(train_ll)),
    }


def sample_nll(fitter, params, data, n_evals):
    """Mean total NLL over n_evals IBS evaluations at fitter.repeats (+ SEM) — a pure
    read of the model at `params`, does not refit. Used to re-evaluate multistart
    candidates before argmin'ing a winner."""
    totals = []
    for _ in range(n_evals):
        nlls, _ = fitter.evaluate(np.asarray(params, dtype=np.float64), data)
        totals.append(float(np.sum(nlls)))
    totals = np.asarray(totals)
    sem = float(totals.std(ddof=1) / np.sqrt(len(totals))) if len(totals) > 1 else float("nan")
    return float(totals.mean()), sem


def select_winner(model_factory, starts, train_data, test_data, n_workers=None,
                   reeval_repeats=REEVAL_REPEATS, n_evals=REEVAL_N_EVALS, verbose=False):
    """Re-evaluate every clean start's params at high IBS repeats, pick the argmin as
    the winner, then evaluate train/test NLL at the winner only (so held-out loss isn't
    contaminated by the noisy selection itself). Returns (winner_index, winner_record,
    train_nll, test_nll)."""
    if not starts:
        raise ValueError("No starts to select a winner from.")
    if n_workers is None:
        n_workers = default_n_workers()

    model = model_factory()
    fitter = MultiThreadedFitter(model, verbose=verbose, n_repeats=reeval_repeats, n_workers=n_workers)
    try:
        for record in starts:
            mean_nll, sem = sample_nll(fitter, record["params"], train_data, n_evals)
            record["train_nll_reeval"] = mean_nll
            record["train_nll_reeval_sem"] = sem
        winner_idx = int(np.argmin([r["train_nll_reeval"] for r in starts]))
        winner_params = np.asarray(starts[winner_idx]["params"], dtype=float)
        train_nll, _ = fitter.evaluate(winner_params, train_data)
        test_nll, _ = fitter.evaluate(winner_params, test_data)
    finally:
        fitter.close()

    return winner_idx, starts[winner_idx], train_nll, test_nll


def default_model_factory(verbose=False):
    return lambda: TreeSearch(verbose=verbose)
