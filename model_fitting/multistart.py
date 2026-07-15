"""Multistart fitting primitives shared by scripts/fit_all.py (interactive
driver), scripts/fit_one_start.py (per-start worker invoked by sbatch), and
scripts/consolidate.py (winner selection from finished starts).

Mirrors monkey_4iar's src/models.py multistart machinery (start_x0 / write_start_json /
validate_starts / re-eval + argmin winner selection), reimplemented here so ninarow
stays self-contained (no reverse dependency on monkey_4iar).
"""

import json
import os
from pathlib import Path

import numpy as np

from tree_search import TreeSearch
from tree_search_fitter import MultiThreadedFitter, BADS_DEFAULTS

# Re-evaluation config for winner selection: a single fit's IBS NLL is too noisy to
# argmin directly (the start that got lucky noise looks best, not the one that's
# actually best), so every start is re-scored at higher IBS repeats before picking.
REEVAL_REPEATS = 50
REEVAL_N_EVALS = 5


def default_n_workers():
    """SLURM_CPUS_PER_TASK if running inside a job allocation, else a small fixed
    default. MultiThreadedFitter's own default (n_workers<=0 -> os.cpu_count()) is NOT
    safe to use directly outside a job: on a shared login/compute node os.cpu_count()
    reports the WHOLE machine's core count, not any per-job allocation."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"].split(",")[0])
    return 6


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


def to_named(values, param_names):
    """Zip an ordered array of values with param_names into a plain dict, in order."""
    return {name: float(v) for name, v in zip(param_names, values)}


def to_array(named_or_array, param_names):
    """Inverse of to_named: accepts either a {name: value} dict (read back in
    param_names order — dicts preserve insertion order, but we look up by name
    rather than trust it) or a plain array (legacy records written before params
    were named), and returns a plain float array in param_names order."""
    if isinstance(named_or_array, dict):
        return np.asarray([named_or_array[name] for name in param_names], dtype=float)
    return np.asarray(named_or_array, dtype=float)


def write_start_json(path, record):
    """Persist one multistart's result atomically with a success marker, so a start
    file's mere existence means "this start finished cleanly" (tmp -> fsync -> rename)."""
    _atomic_write_json(path, {**record, "status": "ok"})


def write_result_json(path, held_out_index, winner, train_nll, test_nll, n_starts, param_names):
    """Persist the winning fit for one (participant, held-out fold) grid point.

    params/x0 are written as {param_name: value} dicts (not bare arrays) so the JSON
    is self-describing. train_nll is collapsed to a scalar (sum over training trials);
    test_nll is kept as the full per-trial array (needed for downstream bootstrapping
    over held-out trials, not just its sum).
    """
    record = {
        "held_out_index": held_out_index,
        "winning_start": winner["start"],
        "n_starts": n_starts,
        "params": to_named(to_array(winner["params"], param_names), param_names),
        "x0": to_named(to_array(winner["x0"], param_names), param_names),
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
    raw_params = record.get("params", [])
    values = list(raw_params.values()) if isinstance(raw_params, dict) else raw_params
    params = np.asarray(values, dtype=float)
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


def fit_one_start(model, train_data, start, seed=0, n_workers=None, n_repeats=50, verbose=False,
                   bads_options_overrides=None):
    """Run a single BADS start from start_x0(start, model, seed). Returns a start record
    (x0, params, train_nll_raw) — the same schema write_start_json persists.

    bads_options_overrides layers on top of BADS_DEFAULTS (e.g. {"max_fun_evals": 300}
    to cap a smoke-test run) — production callers never pass this, so behavior there
    is unchanged."""
    if n_workers is None:
        n_workers = default_n_workers()
    x0 = start_x0(start, model, seed)
    model.initial_params = x0
    fitter = MultiThreadedFitter(model, verbose=verbose, n_repeats=n_repeats, n_workers=n_workers)
    bads_options = {**BADS_DEFAULTS, "display": "iter" if verbose else "off"}
    if bads_options_overrides:
        bads_options.update(bads_options_overrides)
    try:
        params, train_nll = fitter.fit(train_data, bads_options=bads_options)
    finally:
        fitter.close()
    return {
        "start": start,
        "x0": to_named(x0, model.param_names),
        "params": to_named(params, model.param_names),
        "train_nll_raw": float(np.sum(train_nll)),
    }


def sample_nll(fitter, params, param_names, data, n_evals):
    """Mean total NLL over n_evals IBS evaluations at fitter.repeats (+ SEM) — a pure
    read of the model at `params` (a {name: value} dict or plain array), does not
    refit. Used to re-evaluate multistart candidates before argmin'ing a winner."""
    params_arr = to_array(params, param_names)
    totals = []
    for _ in range(n_evals):
        nlls, _ = fitter.evaluate(params_arr, data)
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
            mean_nll, sem = sample_nll(fitter, record["params"], model.param_names, train_data, n_evals)
            record["train_nll_reeval"] = mean_nll
            record["train_nll_reeval_sem"] = sem
        winner_idx = int(np.argmin([r["train_nll_reeval"] for r in starts]))
        winner_params = to_array(starts[winner_idx]["params"], model.param_names)
        train_nll, _ = fitter.evaluate(winner_params, train_data)
        test_nll, _ = fitter.evaluate(winner_params, test_data)
    finally:
        fitter.close()

    return winner_idx, starts[winner_idx], train_nll, test_nll


def default_model_factory(verbose=False, exclude_feature_drop=False):
    """exclude_feature_drop=True mirrors monkey_4iar's frozen-feature_drop regime:
    drops it from the fitted parameter vector entirely (set_params re-inserts 0.0
    at its canonical control index) rather than merely pinning it to a narrow
    plausible range while still letting BADS search it as a free dimension."""
    return lambda: TreeSearch(verbose=verbose, exclude_feature_drop=exclude_feature_drop)
