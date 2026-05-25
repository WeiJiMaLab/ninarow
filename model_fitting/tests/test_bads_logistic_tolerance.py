"""
One fast sanity check: production BADS tols vs scipy on a tiny logistic split.

Do NOT run the full file in agent loops. Prefer reporting results from
``python -m logistic_smoke`` (one split, ~1 min).

    pytest tests/test_bads_logistic_tolerance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_MODEL_FITTING = Path(__file__).resolve().parent.parent
if str(_MODEL_FITTING) not in sys.path:
    sys.path.insert(0, str(_MODEL_FITTING))

from logistic_smoke import (  # noqa: E402
    BASELINE_SCIPY_FTOL,
    BASELINE_SCIPY_GTOL,
    PRODUCTION_TOL_FUN,
    PRODUCTION_TOL_MESH,
    SMOKE_MAX_FUN_EVALS,
    _HAS_PYBADS,
    fit_bads,
    fit_reference,
    fit_scipy,
    get_bounds,
    logistic_nll,
    make_smoke_dataset,
)


def test_production_bads_and_scipy_baseline_finite():
    """Single train split only; capped BADS evals."""
    if not _HAS_PYBADS:
        pytest.skip("pybads not installed")

    X, y = make_smoke_dataset(n_samples=100, n_features=4)
    n_train = 70
    X_train, y_train = X[:n_train], y[:n_train]
    n = X.shape[1]
    lb, ub, plb, pub = get_bounds(n)
    x0 = np.zeros(n)

    ref = fit_reference(X_train, y_train, lb=lb, ub=ub)
    bads = fit_bads(
        X_train,
        y_train,
        x0,
        lb,
        ub,
        plb,
        pub,
        tol_mesh=PRODUCTION_TOL_MESH,
        tol_fun=PRODUCTION_TOL_FUN,
        max_fun_evals=SMOKE_MAX_FUN_EVALS,
    )
    scipy_fit = fit_scipy(
        X_train,
        y_train,
        x0,
        lb,
        ub,
        ftol=BASELINE_SCIPY_FTOL,
        gtol=BASELINE_SCIPY_GTOL,
    )

    assert np.all(np.isfinite(bads.beta))
    assert np.all(np.isfinite(scipy_fit.beta))
    assert np.isfinite(logistic_nll(bads.beta, X_train, y_train))
    assert np.linalg.norm(scipy_fit.beta - ref.beta) < 0.1
