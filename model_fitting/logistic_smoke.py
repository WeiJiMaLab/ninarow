"""
PyBADS vs sklearn/scipy logistic regression smoke test.

Deterministic binary logistic NLL on a **tiny** synthetic dataset. Purpose: check
that production ``tol_mesh`` / ``tol_fun`` land near sensible scipy/sklearn stops
before running any large grid or real fits.

Run (from ``model_fitting/``) — should finish in **under ~2 minutes**::

    python -m logistic_smoke              # default: tiny data, 2-fold, capped BADS evals
    pytest tests/test_bads_logistic_tolerance.py -v

Benchmark only (NOT a smoke test; can take hours)::

    python -m logistic_smoke --benchmark [--quick]

Suggested fair logistic baseline (for later work; from tiny-data smoke)::

- PyBADS (fitter): ``tol_mesh=1e-3``, ``tol_fun=1e-7``
- scipy L-BFGS-B: ``ftol=gtol=1e-6``
- sklearn lbfgs: ``tol=1e-6``, ``C=1e12``

There is no exact analytic map between mesh/stall stops and gradient stops; see
``ninarow/AGENTS.md`` (Smoke tests).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.optimize import Bounds, minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from pybads import BADS

    _HAS_PYBADS = True
except ImportError:
    BADS = None  # type: ignore[misc, assignment]
    _HAS_PYBADS = False

try:
    from sklearn.linear_model import LogisticRegression

    _HAS_SKLEARN_LR = True
except ImportError:
    LogisticRegression = None  # type: ignore[misc, assignment]
    _HAS_SKLEARN_LR = False

# Production BADS defaults (monkey_4iar TreeSearchRunner.DEFAULT_BADS_OPTIONS)
PRODUCTION_TOL_MESH = 1e-3
PRODUCTION_TOL_FUN = 1e-7
FITTER_MAX_FUN_EVALS = 2000
FITTER_FUN_EVAL_START = 20

# Smoke-test budget (agents: do NOT use production max_fun_evals / n=2000 here)
SMOKE_N_SAMPLES = 150
SMOKE_N_FEATURES = 4
SMOKE_N_SPLITS = 2
SMOKE_MAX_FUN_EVALS = 60
SMOKE_FUN_EVAL_START = 8

# Classical baseline to compare against production BADS tols (tiny-data smoke)
BASELINE_SCIPY_FTOL = 1e-6
BASELINE_SCIPY_GTOL = 1e-6
BASELINE_SKLEARN_TOL = 1e-6

OUTPUT_DIR = Path(__file__).resolve().parent / "tests" / "output"


@dataclass(frozen=True)
class ToleranceSetting:
    method: str
    label: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitResult:
    beta: np.ndarray
    nll_train: float
    n_evals: int | None = None
    stop_reason: str = ""
    mesh_size: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def logistic_nll(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for binary logistic regression (sum over trials)."""
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def make_smoke_dataset(
    n_samples: int = SMOKE_N_SAMPLES,
    n_features: int = SMOKE_N_FEATURES,
    n_informative: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary classification with intercept column prepended."""
    if n_informative is None:
        n_informative = max(2, n_features - 1)
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.hstack([np.ones((n_samples, 1)), X_scaled])
    return X.astype(np.float64), y.astype(np.float64)


def get_bounds(n_params: int, bound: float = 5.0, plausible_frac: float = 0.8) -> tuple[np.ndarray, ...]:
    """Box and plausible bounds (inner fraction of box), matching TreeSearch style."""
    lb = np.full(n_params, -bound, dtype=np.float64)
    ub = np.full(n_params, bound, dtype=np.float64)
    margin = bound * (1.0 - plausible_frac)
    plb = lb + margin
    pub = ub - margin
    return lb, ub, plb, pub


def fit_reference(
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
) -> FitResult:
    """Tight scipy L-BFGS-B reference solution."""
    n = X.shape[1]
    if lb is None or ub is None:
        lb, ub, _, _ = get_bounds(n)
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    def objective(beta: np.ndarray) -> float:
        return logistic_nll(beta, X, y)

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=Bounds(lb, ub, keep_feasible=True),
        options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 10_000},
    )
    beta = np.asarray(res.x, dtype=np.float64)
    return FitResult(
        beta=beta,
        nll_train=logistic_nll(beta, X, y),
        n_evals=int(res.nfev),
        stop_reason=str(res.message),
    )


def fit_scipy(
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    ftol: float = 1e-6,
    gtol: float = 1e-6,
) -> FitResult:
    def objective(beta: np.ndarray) -> float:
        return logistic_nll(beta, X, y)

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=Bounds(lb, ub, keep_feasible=True),
        options={"ftol": ftol, "gtol": gtol, "maxiter": 10_000},
    )
    beta = np.asarray(res.x, dtype=np.float64)
    return FitResult(
        beta=beta,
        nll_train=logistic_nll(beta, X, y),
        n_evals=int(res.nfev),
        stop_reason=str(res.message),
        extra={"ftol": ftol, "gtol": gtol},
    )


def fit_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-6,
    C: float = 1e12,
    max_iter: int = 5000,
) -> FitResult:
    if not _HAS_SKLEARN_LR:
        raise ImportError("sklearn.linear_model.LogisticRegression is required")

    # X already includes intercept column; disable separate intercept
    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        tol=tol,
        fit_intercept=False,
        max_iter=max_iter,
        warm_start=False,
    )
    clf.fit(X, y)
    beta = np.asarray(clf.coef_.ravel(), dtype=np.float64)
    return FitResult(
        beta=beta,
        nll_train=logistic_nll(beta, X, y),
        n_evals=getattr(clf, "n_iter_", None),
        stop_reason="sklearn_lbfgs",
        extra={"tol": tol, "C": C},
    )


def fit_bads(
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    plb: np.ndarray,
    pub: np.ndarray,
    tol_mesh: float = PRODUCTION_TOL_MESH,
    tol_fun: float = PRODUCTION_TOL_FUN,
    max_fun_evals: int = SMOKE_MAX_FUN_EVALS,
    fun_eval_start: int = SMOKE_FUN_EVAL_START,
    display: str = "off",
) -> FitResult:
    if not _HAS_PYBADS:
        raise ImportError("pybads is required for BADS fits")

    def objective(beta: np.ndarray) -> float:
        return logistic_nll(np.asarray(beta, dtype=np.float64).ravel(), X, y)

    options = {
        "uncertainty_handling": False,
        "specify_target_noise": False,
        "tol_mesh": tol_mesh,
        "tol_fun": tol_fun,
        "fun_eval_start": fun_eval_start,
        "max_fun_evals": max_fun_evals,
        "display": display,
    }
    opt = BADS(
        objective,
        x0,
        lb,
        ub,
        plb,
        pub,
        options=options,
    )
    result = opt.optimize()
    beta = np.asarray(result["x"], dtype=np.float64).ravel()
    mesh_size = result.get("mesh_size")
    if hasattr(mesh_size, "item"):
        mesh_size = float(mesh_size.item())
    elif mesh_size is not None:
        mesh_size = float(mesh_size)

    return FitResult(
        beta=beta,
        nll_train=logistic_nll(beta, X, y),
        n_evals=int(result.get("func_count", 0)),
        stop_reason=str(result.get("message", result.get("termination_msg", ""))),
        mesh_size=mesh_size,
        extra={"tol_mesh": tol_mesh, "tol_fun": tol_fun},
    )


def _fit_from_setting(
    setting: ToleranceSetting,
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    plb: np.ndarray,
    pub: np.ndarray,
    max_fun_evals: int,
) -> FitResult:
    if setting.method == "scipy":
        return fit_scipy(X, y, x0, lb, ub, **setting.kwargs)
    if setting.method == "sklearn":
        return fit_sklearn(X, y, **setting.kwargs)
    if setting.method == "bads":
        return fit_bads(
            X, y, x0, lb, ub, plb, pub, max_fun_evals=max_fun_evals, **setting.kwargs
        )
    raise ValueError(f"Unknown method: {setting.method}")


def iter_tolerance_settings(quick: bool = False) -> Iterator[ToleranceSetting]:
    """Yield (method, tolerance) combinations for the grid search."""
    if quick:
        bads_mesh = [1e-2, 1e-3, 1e-4]
        bads_fun = [1e-3, 1e-7]
        scipy_tols = [1e-4, 1e-6, 1e-8]
        sklearn_tols = [1e-4, 1e-6, 1e-8]
    else:
        bads_mesh = [1e-2, 1e-3, 1e-4]
        bads_fun = [1e-3, 1e-5, 1e-7]
        scipy_tols = [1e-4, 1e-6, 1e-8]
        sklearn_tols = [1e-4, 1e-6, 1e-8]

    for tol_mesh in bads_mesh:
        for tol_fun in bads_fun:
            yield ToleranceSetting(
                "bads",
                f"bads_mesh={tol_mesh:g}_fun={tol_fun:g}",
                {"tol_mesh": tol_mesh, "tol_fun": tol_fun},
            )

    for tol in scipy_tols:
        yield ToleranceSetting(
            "scipy",
            f"scipy_ftol={tol:g}_gtol={tol:g}",
            {"ftol": tol, "gtol": tol},
        )

    for tol in sklearn_tols:
        yield ToleranceSetting(
            "sklearn",
            f"sklearn_tol={tol:g}",
            {"tol": tol},
        )


@dataclass
class CVRow:
    setting: ToleranceSetting
    cv_nll_mean: float
    cv_nll_std: float
    delta_nll_vs_ref: float
    param_l2_vs_ref: float
    n_evals_mean: float
    fold_nlls: list[float] = field(default_factory=list)


def cross_validate_setting(
    X: np.ndarray,
    y: np.ndarray,
    setting: ToleranceSetting,
    n_splits: int = 5,
    random_state: int = 0,
    max_fun_evals: int = 2000,
) -> CVRow:
    """Fit per fold; compare held-out NLL and params to per-fold scipy reference."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_params = X.shape[1]
    lb, ub, plb, pub = get_bounds(n_params)

    fold_test_nlls: list[float] = []
    fold_delta_nll: list[float] = []
    fold_param_l2: list[float] = []
    fold_n_evals: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        ref = fit_reference(X_train, y_train, lb=lb, ub=ub)
        x0 = np.zeros(n_params, dtype=np.float64)

        fit = _fit_from_setting(setting, X_train, y_train, x0, lb, ub, plb, pub, max_fun_evals)

        test_nll = logistic_nll(fit.beta, X_test, y_test)
        ref_test_nll = logistic_nll(ref.beta, X_test, y_test)

        fold_test_nlls.append(test_nll)
        fold_delta_nll.append(test_nll - ref_test_nll)
        fold_param_l2.append(float(np.linalg.norm(fit.beta - ref.beta)))
        if fit.n_evals is not None:
            ne = fit.n_evals
            fold_n_evals.append(float(ne.item() if hasattr(ne, "item") else ne))

    return CVRow(
        setting=setting,
        cv_nll_mean=float(np.mean(fold_test_nlls)),
        cv_nll_std=float(np.std(fold_test_nlls)),
        delta_nll_vs_ref=float(np.mean(fold_delta_nll)),
        param_l2_vs_ref=float(np.mean(fold_param_l2)),
        n_evals_mean=float(np.mean(fold_n_evals)) if fold_n_evals else float("nan"),
        fold_nlls=fold_test_nlls,
    )


def run_grid(
    quick: bool = False,
    n_splits: int | None = None,
    max_fun_evals: int = 2000,
    save_plots: bool = True,
    save_csv: bool = True,
) -> list[CVRow]:
    """Run full tolerance grid with cross-validation."""
    if n_splits is None:
        n_splits = 3 if quick else 5

    X, y = make_smoke_dataset()
    settings = list(iter_tolerance_settings(quick=quick))
    rows: list[CVRow] = []

    print(f"Logistic smoke test: {len(settings)} settings, {n_splits}-fold CV, n={len(y)}")
    print("-" * 100)
    print(
        f"{'method':<8} {'label':<32} {'cv_nll':>10} {'d_nll':>10} "
        f"{'param_l2':>10} {'n_evals':>10}"
    )
    print("-" * 100)

    for setting in settings:
        row = cross_validate_setting(
            X, y, setting, n_splits=n_splits, max_fun_evals=max_fun_evals
        )
        rows.append(row)
        print(
            f"{setting.method:<8} {setting.label:<32} "
            f"{row.cv_nll_mean:10.4f} {row.delta_nll_vs_ref:+10.4f} "
            f"{row.param_l2_vs_ref:10.4f} {row.n_evals_mean:10.1f}"
        )

    print("-" * 100)
    _print_empirical_pairs(rows)

    if save_plots:
        _save_plots(rows, quick=quick)
    if save_csv:
        _save_csv(rows, quick=quick)

    return rows


def _print_empirical_pairs(rows: list[CVRow]) -> None:
    """Summarize closest scipy/sklearn match to production BADS."""
    prod = [
        r
        for r in rows
        if r.setting.method == "bads"
        and r.setting.kwargs.get("tol_mesh") == PRODUCTION_TOL_MESH
        and r.setting.kwargs.get("tol_fun") == PRODUCTION_TOL_FUN
    ]
    if not prod:
        return
    prod_row = prod[0]
    print("\nProduction-like BADS:", prod_row.setting.label)
    print(f"  cv_nll={prod_row.cv_nll_mean:.4f}, d_nll={prod_row.delta_nll_vs_ref:+.4f}, "
          f"param_l2={prod_row.param_l2_vs_ref:.4f}")

    others = [r for r in rows if r.setting.method in ("scipy", "sklearn")]
    others.sort(key=lambda r: abs(r.delta_nll_vs_ref) + r.param_l2_vs_ref)
    if others:
        best = others[0]
        print(
            f"  Closest classical: {best.setting.label} "
            f"(d_nll={best.delta_nll_vs_ref:+.4f}, param_l2={best.param_l2_vs_ref:.4f})"
        )


def _save_csv(rows: list[CVRow], quick: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / ("logistic_smoke_quick.csv" if quick else "logistic_smoke.csv")
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "label",
                "kwargs",
                "cv_nll_mean",
                "cv_nll_std",
                "delta_nll_vs_ref",
                "param_l2_vs_ref",
                "n_evals_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.setting.method,
                    "label": row.setting.label,
                    "kwargs": json.dumps(row.setting.kwargs),
                    "cv_nll_mean": row.cv_nll_mean,
                    "cv_nll_std": row.cv_nll_std,
                    "delta_nll_vs_ref": row.delta_nll_vs_ref,
                    "param_l2_vs_ref": row.param_l2_vs_ref,
                    "n_evals_mean": row.n_evals_mean,
                }
            )
    print(f"\nWrote {path}")


def _save_plots(rows: list[CVRow], quick: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "logistic_smoke_quick" if quick else "logistic_smoke"

    methods = sorted({r.setting.method for r in rows})
    colors = {"bads": "C0", "scipy": "C1", "sklearn": "C2"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        c = colors.get(row.setting.method, "gray")
        ax.scatter(
            row.param_l2_vs_ref,
            row.delta_nll_vs_ref,
            c=c,
            alpha=0.7,
            s=40,
            label=row.setting.method if row.setting.method not in ax.get_legend_handles_labels()[1] else "",
        )
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("param L2 vs per-fold reference")
    ax.set_ylabel("delta CV test NLL vs reference")
    ax.set_title("Optimizer tolerance comparison (logistic smoke)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], label=m) for m in methods]
    ax.legend(handles=handles)
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / f"{prefix}_scatter.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {scatter_path}")

    bads_rows = [r for r in rows if r.setting.method == "bads"]
    if bads_rows:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        tol_fun_fixed = PRODUCTION_TOL_FUN
        subset = [
            r
            for r in bads_rows
            if r.setting.kwargs.get("tol_fun") == tol_fun_fixed
        ]
        if subset:
            meshes = sorted({r.setting.kwargs["tol_mesh"] for r in subset})
            means = []
            for m in meshes:
                match = [r for r in subset if r.setting.kwargs["tol_mesh"] == m]
                means.append(match[0].cv_nll_mean if match else np.nan)
            ax2.plot(meshes, means, "o-")
            ax2.set_xscale("log")
            ax2.set_xlabel("tol_mesh")
            ax2.set_ylabel("mean CV test NLL")
            ax2.set_title(f"BADS CV NLL vs tol_mesh (tol_fun={tol_fun_fixed:g})")
            fig2.tight_layout()
            heat_path = OUTPUT_DIR / f"{prefix}_bads_mesh.png"
            fig2.savefig(heat_path, dpi=120)
            plt.close(fig2)
            print(f"Wrote {heat_path}")


def iter_smoke_tolerance_settings() -> Iterator[ToleranceSetting]:
    """Small set of tols for fast smoke (production BADS + nearby + baseline classical)."""
    yield ToleranceSetting(
        "bads",
        "prod_tol_mesh=1e-3_tol_fun=1e-7",
        {"tol_mesh": PRODUCTION_TOL_MESH, "tol_fun": PRODUCTION_TOL_FUN},
    )
    yield ToleranceSetting(
        "bads",
        "mesh_1e-4_tol_fun=1e-7",
        {"tol_mesh": 1e-4, "tol_fun": PRODUCTION_TOL_FUN},
    )
    yield ToleranceSetting(
        "bads",
        "mesh_1e-2_tol_fun=1e-7",
        {"tol_mesh": 1e-2, "tol_fun": PRODUCTION_TOL_FUN},
    )
    yield ToleranceSetting(
        "scipy",
        f"baseline_ftol={BASELINE_SCIPY_FTOL:g}",
        {"ftol": BASELINE_SCIPY_FTOL, "gtol": BASELINE_SCIPY_GTOL},
    )
    yield ToleranceSetting(
        "sklearn",
        f"baseline_tol={BASELINE_SKLEARN_TOL:g}",
        {"tol": BASELINE_SKLEARN_TOL},
    )


@dataclass
class SplitRow:
    setting: ToleranceSetting
    test_nll: float
    delta_nll_vs_ref: float
    param_l2_vs_ref: float
    n_evals: float


def run_smoke_tols(
    n_samples: int = SMOKE_N_SAMPLES,
    max_fun_evals: int = SMOKE_MAX_FUN_EVALS,
    train_frac: float = 0.7,
) -> list[SplitRow]:
    """
    Fast smoke: one train/test split, few settings, capped BADS evals. Target < 2 min.
    """
    X, y = make_smoke_dataset(n_samples=n_samples)
    n_train = int(train_frac * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    n_params = X.shape[1]
    lb, ub, plb, pub = get_bounds(n_params)
    x0 = np.zeros(n_params, dtype=np.float64)

    ref = fit_reference(X_train, y_train, lb=lb, ub=ub)
    settings = list(iter_smoke_tolerance_settings())
    rows: list[SplitRow] = []

    print(
        f"SMOKE (one split, target <2 min): n={n_samples}, train={n_train}, "
        f"BADS max_fun_evals={max_fun_evals}"
    )
    print(
        f"Fitter tols: tol_mesh={PRODUCTION_TOL_MESH}, tol_fun={PRODUCTION_TOL_FUN}"
    )
    print(
        f"Fair logistic baseline (later): scipy ftol=gtol={BASELINE_SCIPY_GTOL:g}"
    )
    print("-" * 90)
    print(f"{'label':<32} {'test_nll':>9} {'d_nll':>9} {'param_l2':>9} {'n_evals':>8}")
    print("-" * 90)

    for setting in settings:
        fit = _fit_from_setting(
            setting, X_train, y_train, x0, lb, ub, plb, pub, max_fun_evals
        )
        test_nll = logistic_nll(fit.beta, X_test, y_test)
        ref_test = logistic_nll(ref.beta, X_test, y_test)
        ne = fit.n_evals
        n_evals = float(ne.item() if ne is not None and hasattr(ne, "item") else (ne or 0))
        row = SplitRow(
            setting=setting,
            test_nll=test_nll,
            delta_nll_vs_ref=test_nll - ref_test,
            param_l2_vs_ref=float(np.linalg.norm(fit.beta - ref.beta)),
            n_evals=n_evals,
        )
        rows.append(row)
        print(
            f"{setting.label:<32} {row.test_nll:9.3f} "
            f"{row.delta_nll_vs_ref:+9.4f} {row.param_l2_vs_ref:9.4f} "
            f"{row.n_evals:8.0f}"
        )

    print("-" * 90)
    bads = [r for r in rows if r.setting.method == "bads"]
    if len(bads) >= 2:
        d_spread = max(r.delta_nll_vs_ref for r in bads) - min(
            r.delta_nll_vs_ref for r in bads
        )
        l2_spread = max(r.param_l2_vs_ref for r in bads) - min(
            r.param_l2_vs_ref for r in bads
        )
        print(
            f"BADS tol_mesh spread (tol_fun={PRODUCTION_TOL_FUN:g}): "
            f"Δd_nll={d_spread:.4f}, Δparam_l2={l2_spread:.4f}"
        )
    prod = next(
        (r for r in bads if r.setting.kwargs.get("tol_mesh") == PRODUCTION_TOL_MESH),
        None,
    )
    scipy_rows = [r for r in rows if r.setting.method == "scipy"]
    if prod and scipy_rows:
        print(
            f"Production BADS vs scipy baseline: "
            f"Δd_nll diff={abs(prod.delta_nll_vs_ref - scipy_rows[0].delta_nll_vs_ref):.4f}"
        )
    return rows


def iter_fitter_tolerance_settings() -> Iterator[ToleranceSetting]:
    """Current TreeSearchRunner / MultiThreadedFitter BADS tols + classical comparators."""
    yield ToleranceSetting(
        "bads",
        f"fitter_tol_mesh={PRODUCTION_TOL_MESH:g}_tol_fun={PRODUCTION_TOL_FUN:g}",
        {
            "tol_mesh": PRODUCTION_TOL_MESH,
            "tol_fun": PRODUCTION_TOL_FUN,
            "fun_eval_start": FITTER_FUN_EVAL_START,
        },
    )
    for tol in (1e-4, 1e-6, 1e-8):
        yield ToleranceSetting(
            "scipy",
            f"scipy_ftol={tol:g}_gtol={tol:g}",
            {"ftol": tol, "gtol": tol},
        )
    for tol in (1e-4, 1e-6, 1e-8):
        yield ToleranceSetting(
            "sklearn",
            f"sklearn_tol={tol:g}",
            {"tol": tol},
        )


def run_fitter_tols(
    n_splits: int = SMOKE_N_SPLITS,
    max_fun_evals: int = SMOKE_MAX_FUN_EVALS,
    save_plots: bool = False,
    save_csv: bool = False,
) -> list[CVRow]:
    """Same as run_smoke_tols but includes extra scipy/sklearn tol grid (still tiny data)."""
    X, y = make_smoke_dataset()
    settings = list(iter_fitter_tolerance_settings())
    rows: list[CVRow] = []

    print(
        "Fitter production BADS tols: "
        f"tol_mesh={PRODUCTION_TOL_MESH}, tol_fun={PRODUCTION_TOL_FUN}, "
        f"fun_eval_start={FITTER_FUN_EVAL_START}, max_fun_evals={max_fun_evals}"
    )
    print(f"{len(settings)} settings, {n_splits}-fold CV, n={len(y)}")
    print("-" * 100)
    print(
        f"{'method':<8} {'label':<40} {'cv_nll':>10} {'d_nll':>10} "
        f"{'param_l2':>10} {'n_evals':>10}"
    )
    print("-" * 100)

    for setting in settings:
        row = cross_validate_setting(
            X, y, setting, n_splits=n_splits, max_fun_evals=max_fun_evals
        )
        rows.append(row)
        print(
            f"{setting.method:<8} {setting.label:<40} "
            f"{row.cv_nll_mean:10.4f} {row.delta_nll_vs_ref:+10.4f} "
            f"{row.param_l2_vs_ref:10.4f} {row.n_evals_mean:10.1f}"
        )

    print("-" * 100)
    _print_empirical_pairs(rows)

    if save_plots:
        _save_plots(rows, quick=False)
    if save_csv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "logistic_smoke_fitter_tols.csv"
        import csv

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "label",
                    "kwargs",
                    "cv_nll_mean",
                    "cv_nll_std",
                    "delta_nll_vs_ref",
                    "param_l2_vs_ref",
                    "n_evals_mean",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "method": row.setting.method,
                        "label": row.setting.label,
                        "kwargs": json.dumps(row.setting.kwargs),
                        "cv_nll_mean": row.cv_nll_mean,
                        "cv_nll_std": row.cv_nll_std,
                        "delta_nll_vs_ref": row.delta_nll_vs_ref,
                        "param_l2_vs_ref": row.param_l2_vs_ref,
                        "n_evals_mean": row.n_evals_mean,
                    }
                )
        print(f"\nWrote {path}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBADS vs scipy/sklearn logistic smoke test")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Large tolerance grid on n=2000 (slow; not a smoke test)",
    )
    parser.add_argument(
        "--fitter",
        action="store_true",
        help="Extended tol comparison on tiny data (still capped evals)",
    )
    parser.add_argument("--quick", action="store_true", help="With --benchmark: smaller grid")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--max-fun-evals", type=int, default=SMOKE_MAX_FUN_EVALS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    if not _HAS_PYBADS:
        raise SystemExit("pybads is not installed; cannot run BADS smoke test")

    if args.benchmark:
        run_grid(
            quick=args.quick,
            n_splits=args.n_splits,
            max_fun_evals=args.max_fun_evals if args.max_fun_evals != SMOKE_MAX_FUN_EVALS else FITTER_MAX_FUN_EVALS,
            save_plots=not args.no_plots,
            save_csv=not args.no_csv,
        )
        return

    if args.fitter:
        run_fitter_tols(
            n_splits=args.n_splits or SMOKE_N_SPLITS,
            max_fun_evals=args.max_fun_evals,
            save_plots=not args.no_plots,
            save_csv=not args.no_csv,
        )
        return

    run_smoke_tols(max_fun_evals=args.max_fun_evals)


if __name__ == "__main__":
    main()
