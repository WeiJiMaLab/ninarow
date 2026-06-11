"""Shared definitions for the TreeSearch parameter-recovery experiment.

Parameter recovery asks: if we *simulate* choices from the model at a known
parameter vector theta_true and then refit with the standard pipeline, do we get
theta_true back? Because ``TreeSearch.predict`` samples stochastically from the
model's choice distribution (the heuristic carries a seeded RNG, and lapse /
feature_drop / pruning / stopping are all RNG-driven), one ``predict`` call is one
draw from the model -- the same primitive IBS uses to estimate the likelihood. So
recovery is self-consistent by construction.

This module pins the model definition (the full 6-template set, including 1IAR)
and the BADS options so that generation and refitting use identical settings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling ninarow model_fitting importable regardless of CWD.
_MODEL_FITTING = Path(__file__).resolve().parent.parent
if str(_MODEL_FITTING) not in sys.path:
    sys.path.insert(0, str(_MODEL_FITTING))

from tree_search import TreeSearch, feature_list_from_templates  # noqa: E402

# Full 6-group heuristic template set (matches monkey_4iar production fit.py).
# 1IAR is the single-piece group with no existing MLE fit; its ground-truth weight
# is seeded from a range around 2IAR_DIS (see export_ground_truth.py).
SIX_TEMPLATES = {
    "4IAR": [[1, 1, 1, 1]],
    "3IAR_CON": [[0, 1, 1, 1], [1, 1, 1, 0]],
    "3IAR_DIS": [[1, 0, 1, 1], [1, 1, 0, 1]],
    "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    "1IAR": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
}

# Mirrors monkey_4iar TreeSearchRunner so recovery fits behave like production.
DEFAULT_BADS_OPTIONS = {
    "uncertainty_handling": True,
    "specify_target_noise": True,
    "noise_final_samples": 0,
    "tol_fun": 1e-7,
    "tol_mesh": 1e-3,
    "fun_eval_start": 20,
    "max_fun_evals": 10000,
}
DEFAULT_ATOL_MESH = 5e-3
DEFAULT_ATOL_FUN = 1e-7
DEFAULT_N_REPEATS = 60


def build_model(verbose=False):
    """Construct a fresh 6-template TreeSearch with the canonical parameter order."""
    return TreeSearch(feature_list=feature_list_from_templates(SIX_TEMPLATES), verbose=verbose)


def param_names(model=None):
    """Ordered parameter names for the 6-template model (len 12)."""
    if model is None:
        model = build_model()
    return list(model.param_names)


def params_dict_to_vector(values, model=None):
    """Map a {param_name: value} dict to the model's ordered parameter vector."""
    names = param_names(model)
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"Missing parameters: {sorted(missing)}")
    return np.array([float(values[n]) for n in names], dtype=np.float64)


def params_vector_to_dict(vector, model=None):
    """Inverse of params_dict_to_vector."""
    names = param_names(model)
    if len(vector) != len(names):
        raise ValueError(f"Expected {len(names)} params, got {len(vector)}")
    return {n: float(v) for n, v in zip(names, vector)}


def load_boards(path, max_trials=None, seed=0):
    """Load board positions (black/white/color/...) from a CSV file or a directory
    of fold CSVs. The original ``move`` column is discarded -- only board state and
    side-to-move are used; synthetic moves are sampled in generate.py.
    """
    path = Path(path)
    if path.is_dir():
        csvs = sorted(path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSVs under {path}")
        frames = [pd.read_csv(c) for c in csvs]
        boards = pd.concat(frames, ignore_index=True)
    else:
        boards = pd.read_csv(path)

    needed = {"black", "white", "color"}
    missing = needed - set(boards.columns)
    if missing:
        raise ValueError(f"Board source missing columns: {sorted(missing)}")

    if max_trials is not None and len(boards) > max_trials:
        boards = boards.sample(n=max_trials, random_state=seed).reset_index(drop=True)
    return boards.reset_index(drop=True)
