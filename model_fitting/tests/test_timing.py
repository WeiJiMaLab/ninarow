"""
Timing and equivalence for SingleThreadedFitter vs MultiThreadedFitter.

model_fit.ModelFitter is not included here: its IBS / n_repeats path does not
line up with tree_search fitters for arbitrary n_repeats.

(1) Verify SingleThreadedFitter == MultiThreadedFitter(n_workers=1)
(2) Benchmark ST vs MT(1), MT(6), MT(all CPUs) for time
(3) Optional: scan n_repeats and compare ST vs MT ratios (more IBS work per trial).
(4) Optional: scan n_trials (default practical grid: 128, 256, 512; rows tiled if needed).
(5) BADS: one short ``fit()`` (low ``max_fun_evals``); assert ST vs MT(1) match; time ST vs MT(1,6,12).
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
import glob
import random
import time

import numpy as np
import pandas as pd

_MODEL_FITTING = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent


def _load_module_from_path(unique_name: str, file_path: Path):
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(_MODEL_FITTING))

_ts = _load_module_from_path("tree_search", _MODEL_FITTING / "tree_search.py")
_tsf = _load_module_from_path("tree_search_fitter", _MODEL_FITTING / "tree_search_fitter.py")

TreeSearch = _ts.TreeSearch
SingleThreadedFitter = _ts.SingleThreadedFitter
MultiThreadedFitter = _tsf.MultiThreadedFitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATES = {
    "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
    "4IAR": [[1, 1, 1, 1]],
}
WEIGHTS = {"2IAR_CON": 1.0, "2IAR_DIS": 0.4, "3IAR": 3.5, "4IAR": 9.0}

# One BADS run with a small budget (same spirit as timing loops: fast, not converged).
BADS_MINI_OPTIONS = {
    "uncertainty_handling": True,
    "noise_final_samples": 0,
    "max_fun_evals": 1,
}


def load_data(data_folder, fold_idx=0, n_trials=20):
    """
    Load up to ``n_trials`` rows from split CSV(s).

    If ``n_trials`` exceeds the number of rows in the chosen fold, the fold is
    repeated (tiled) so timing runs can use a larger N without extra fixture
    files. ``trial_id`` is reassigned to unique integers after tiling.
    """
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    if not split_files:
        raise ValueError(f"No split files found in {data_folder}")
    data = [pd.read_csv(f) for f in split_files]
    fold_idx = min(fold_idx, len(data) - 1)
    df = data[fold_idx].copy()
    if len(df) == 0:
        raise ValueError("Empty split file")
    if n_trials > len(df):
        k = int(np.ceil(n_trials / len(df)))
        df = pd.concat([df] * k, ignore_index=True)
    df = df.iloc[:n_trials].copy()
    df["trial_id"] = range(len(df))
    return df


def make_tree_search_model(feature_drop=0.0):
    model = TreeSearch(templates=TEMPLATES, initial_weights=WEIGHTS, verbose=False)
    if "feature_drop" in model.param_names:
        idx = model.param_names.index("feature_drop")
        model.initial_params[idx] = feature_drop
        model.lower_bound[idx] = feature_drop
        model.upper_bound[idx] = feature_drop
    return model


def run_tree_search_iterations(fitter, params, n_iterations, manual_seed):
    random.seed(manual_seed)
    _ = random.randint(0, 2**64)
    nlls, times = [], []
    for _ in range(n_iterations):
        start = time.perf_counter()
        nll = fitter.optimize(params)
        times.append(time.perf_counter() - start)
        nlls.append(nll)
    return nlls, times


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def run_timing_test(data_folder=None, n_trials=20, n_iterations=20,
                    manual_seed=1, verbose=True, feature_drop=0.0, n_repeats=1):
    """
    (1) Equivalence: SingleThreaded == MultiThreaded(1) (same n_repeats)
    (2) Timing: SingleThreaded, MultiThreaded(1), MultiThreaded(6), MultiThreaded(-1)
    """
    if data_folder is None:
        data_folder = str(_TESTS_DIR / "data")

    data = load_data(data_folder, n_trials=n_trials)
    data["expected_counts"] = 1
    if "trial_id" not in data.columns:
        data["trial_id"] = range(len(data))

    model_st = make_tree_search_model(feature_drop)
    fitter_st = SingleThreadedFitter(model_st, n_repeats=n_repeats, verbose=False)
    fitter_st.data = data.copy()

    model_mt1 = make_tree_search_model(feature_drop)
    fitter_mt1 = MultiThreadedFitter(
        model_mt1, n_repeats=n_repeats, verbose=False, n_workers=1)
    fitter_mt1.data = data.copy()

    model_mt6 = make_tree_search_model(feature_drop)
    fitter_mt6 = MultiThreadedFitter(
        model_mt6, n_repeats=n_repeats, verbose=False, n_workers=6)
    fitter_mt6.data = data.copy()

    model_mtn = make_tree_search_model(feature_drop)
    fitter_mtn = MultiThreadedFitter(
        model_mtn, n_repeats=n_repeats, verbose=False, n_workers=-1)
    fitter_mtn.data = data.copy()

    test_params = model_st.initial_params.copy()

    if verbose:
        print("=" * 90)
        print(f"PHASE 1 — EQUIVALENCE: SingleThreaded vs MultiThreaded(1), n_repeats={n_repeats}")
        print("=" * 90)

    if verbose:
        print("  Running SingleThreadedFitter ...")
    nll_st, times_st = run_tree_search_iterations(
        fitter_st, test_params, n_iterations, manual_seed)

    if verbose:
        print("  Running MultiThreadedFitter(n_workers=1) ...")
    nll_mt1, times_mt1 = run_tree_search_iterations(
        fitter_mt1, test_params, n_iterations, manual_seed)

    eq_st_mt1 = _compare(nll_st, nll_mt1, "SingleThreaded", "MultiThreaded(1)", verbose)

    if verbose:
        print()
        print("=" * 90)
        print(f"PHASE 2 — TIMING (n_repeats={n_repeats})")
        print("=" * 90)

    if verbose:
        print("  Running MultiThreadedFitter(n_workers=6) ...")
    _, times_mt6 = run_tree_search_iterations(
        fitter_mt6, test_params, n_iterations, manual_seed)

    if verbose:
        print("  Running MultiThreadedFitter(n_workers=-1) ...")
    _, times_mtn = run_tree_search_iterations(
        fitter_mtn, test_params, n_iterations, manual_seed)

    _print_timing_table([
        ("SingleThreadedFitter", times_st),
        ("MultiThreadedFitter(n_workers=1)", times_mt1),
        ("MultiThreadedFitter(n_workers=6)", times_mt6),
        (f"MultiThreadedFitter(n_workers={fitter_mtn.n_workers})", times_mtn),
    ], verbose)

    fitter_mt1.close()
    fitter_mt6.close()
    fitter_mtn.close()

    return eq_st_mt1


def run_bads_fit_equivalence_test(
    data_folder=None,
    n_trials=20,
    manual_seed=1,
    feature_drop=0.0,
    n_repeats=1,
    bads_options=None,
    verbose=True,
):
    """
    Run a single BADS ``fit()`` per fitter (short budget via ``bads_options``).

    Same random seed handshake as :func:`run_tree_search_iterations` before each
    ``fit`` so objective randomness aligns across runs.

    Asserts optimized parameters and per-trial final NLL match SingleThreaded vs
    MultiThreaded for ``n_workers`` in ``(1, 6, 12)``. Prints a timing row per
    configuration (like phase 2 of :func:`run_timing_test`).
    """
    if bads_options is None:
        bads_options = BADS_MINI_OPTIONS
    if data_folder is None:
        data_folder = str(_TESTS_DIR / "data")

    data = load_data(data_folder, n_trials=n_trials)
    data["expected_counts"] = 1
    if "trial_id" not in data.columns:
        data["trial_id"] = range(len(data))

    specs = [
        ("SingleThreadedFitter", "st", None),
        ("MultiThreadedFitter(n_workers=1)", "mt", 1),
        ("MultiThreadedFitter(n_workers=6)", "mt", 6),
        ("MultiThreadedFitter(n_workers=12)", "mt", 12),
    ]

    rows = []
    for label, kind, n_workers in specs:
        random.seed(manual_seed)
        _ = random.randint(0, 2**64)

        model = make_tree_search_model(feature_drop)
        if kind == "st":
            fitter = SingleThreadedFitter(model, n_repeats=n_repeats, verbose=False)
        else:
            fitter = MultiThreadedFitter(
                model, n_repeats=n_repeats, verbose=False, n_workers=n_workers
            )

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            params, final_ll = fitter.fit(data, bads_options=bads_options)
        elapsed = time.perf_counter() - t0

        if kind == "mt":
            fitter.close()

        rows.append((label, params, np.asarray(final_ll, dtype=float), elapsed))

    ref_params = rows[0][1]
    ref_ll = rows[0][2]
    tol = 1e-5

    for label, params, final_ll, _ in rows[1:]:
        assert np.allclose(params, ref_params, rtol=tol, atol=tol), (
            f"BADS fit params mismatch: {label} vs SingleThreadedFitter"
        )
        assert np.allclose(final_ll, ref_ll, rtol=tol, atol=tol), (
            f"BADS fit final NLL mismatch: {label} vs SingleThreadedFitter"
        )

    if verbose:
        print()
        print("=" * 90)
        print("BADS fit — wall time per fitter (one short optimization each)")
        print("=" * 90)
        hdr = f"{'Fitter':<45} {'wall (s)':>12}"
        print(hdr)
        print("-" * len(hdr))
        baseline = rows[0][3]
        for label, _, _, elapsed in rows:
            ratio = baseline / elapsed if elapsed > 0 else float("inf")
            print(f"{label:<45} {elapsed:>12.4f}  ({ratio:.2f}x vs ST)")
        print()

    return True  # explicit so ``assert run_bads_fit_equivalence_test(...)`` passes


def benchmark_st_vs_mt_n_repeats(
    data_folder=None,
    n_trials=20,
    n_iterations=5,
    manual_seed=1,
    feature_drop=0.0,
    n_repeats_list=(1, 5, 10, 25, 50),
    verbose=True,
):
    """
    For each n_repeats, measure mean optimize() time for ST vs MT(1) and MT(-1).
    As n_repeats grows, IBS does more predict() work per trial; multiprocessing
    overhead should become a smaller fraction of total time.
    """
    if data_folder is None:
        data_folder = str(_TESTS_DIR / "data")

    data = load_data(data_folder, n_trials=n_trials)
    data["expected_counts"] = 1
    if "trial_id" not in data.columns:
        data["trial_id"] = range(len(data))

    if verbose:
        print("=" * 90)
        print("ST vs MT — mean optimize() time by n_repeats")
        print("=" * 90)
        print(f"  trials={n_trials}, iterations per cell={n_iterations}, seed={manual_seed}")
        print()

    hdr = (
        f"{'n_repeats':>10} | {'ST (ms)':>10} | {'MT1 (ms)':>10} | "
        f"{'MT* (ms)':>10} | {'MT1/ST':>8} | {'MT*/ST':>8}"
    )
    if verbose:
        print(hdr)
        print("-" * len(hdr))

    mt_workers = None

    for n_rep in n_repeats_list:
        m_st = make_tree_search_model(feature_drop)
        f_st = SingleThreadedFitter(m_st, n_repeats=n_rep, verbose=False)
        f_st.data = data.copy()

        m_mt1 = make_tree_search_model(feature_drop)
        f_mt1 = MultiThreadedFitter(m_mt1, n_repeats=n_rep, verbose=False, n_workers=1)
        f_mt1.data = data.copy()

        m_mtn = make_tree_search_model(feature_drop)
        f_mtn = MultiThreadedFitter(m_mtn, n_repeats=n_rep, verbose=False, n_workers=-1)
        f_mtn.data = data.copy()
        mt_workers = f_mtn.n_workers

        params = m_st.initial_params.copy()

        _, t_st = run_tree_search_iterations(f_st, params, n_iterations, manual_seed)
        _, t_mt1 = run_tree_search_iterations(f_mt1, params, n_iterations, manual_seed)
        _, t_mtn = run_tree_search_iterations(f_mtn, params, n_iterations, manual_seed)

        avg_st = np.mean(t_st) * 1000
        avg_mt1 = np.mean(t_mt1) * 1000
        avg_mtn = np.mean(t_mtn) * 1000
        r1 = avg_mt1 / avg_st if avg_st > 0 else float("nan")
        rn = avg_mtn / avg_st if avg_st > 0 else float("nan")

        if verbose:
            print(
                f"{n_rep:>10} | {avg_st:>10.1f} | {avg_mt1:>10.1f} | "
                f"{avg_mtn:>10.1f} | {r1:>7.2f}x | {rn:>7.2f}x"
            )

        f_mt1.close()
        f_mtn.close()

    if verbose:
        print()
        print(
            f"  MT* = MultiThreadedFitter(n_workers={mt_workers}). "
            "Ratios > 1 mean slower than SingleThreaded."
        )
        print()


def benchmark_st_vs_mt_trials(
    data_folder=None,
    n_trials_list=(128, 256, 512),
    n_iterations=5,
    manual_seed=1,
    feature_drop=0.0,
    n_repeats=10,
    verbose=True,
):
    """
    For each dataset size (trials per ``evaluate``), measure mean ``optimize()``
    time for ST vs MT(1), MT(6), and MT(all CPUs). Rows are tiled via
    :func:`load_data` when the CSV is shorter than ``n_trials``.
    """
    if data_folder is None:
        data_folder = str(_TESTS_DIR / "data")

    if verbose:
        print("=" * 110)
        print("ST vs MT — mean optimize() time by dataset size (n_trials)")
        print("=" * 110)
        print(
            f"  n_repeats={n_repeats} (IBS), iterations per cell={n_iterations}, "
            f"seed={manual_seed}"
        )
        print()

    hdr = (
        f"{'n_trials':>8} | {'ST (ms)':>10} | {'MT1 (ms)':>10} | {'MT6 (ms)':>10} | "
        f"{'MT* (ms)':>10} | {'MT1/ST':>8} | {'MT6/ST':>8} | {'MT*/ST':>8}"
    )
    if verbose:
        print(hdr)
        print("-" * len(hdr))

    mt_workers = None

    for nt in n_trials_list:
        data = load_data(data_folder, n_trials=nt)
        data["expected_counts"] = 1

        m_st = make_tree_search_model(feature_drop)
        f_st = SingleThreadedFitter(m_st, n_repeats=n_repeats, verbose=False)
        f_st.data = data.copy()

        m_mt1 = make_tree_search_model(feature_drop)
        f_mt1 = MultiThreadedFitter(
            m_mt1, n_repeats=n_repeats, verbose=False, n_workers=1)
        f_mt1.data = data.copy()

        m_mt6 = make_tree_search_model(feature_drop)
        f_mt6 = MultiThreadedFitter(
            m_mt6, n_repeats=n_repeats, verbose=False, n_workers=6)
        f_mt6.data = data.copy()

        m_mtn = make_tree_search_model(feature_drop)
        f_mtn = MultiThreadedFitter(
            m_mtn, n_repeats=n_repeats, verbose=False, n_workers=-1)
        f_mtn.data = data.copy()
        mt_workers = f_mtn.n_workers

        params = m_st.initial_params.copy()

        _, t_st = run_tree_search_iterations(f_st, params, n_iterations, manual_seed)
        _, t_mt1 = run_tree_search_iterations(f_mt1, params, n_iterations, manual_seed)
        _, t_mt6 = run_tree_search_iterations(f_mt6, params, n_iterations, manual_seed)
        _, t_mtn = run_tree_search_iterations(f_mtn, params, n_iterations, manual_seed)

        avg_st = np.mean(t_st) * 1000
        avg_mt1 = np.mean(t_mt1) * 1000
        avg_mt6 = np.mean(t_mt6) * 1000
        avg_mtn = np.mean(t_mtn) * 1000
        r1 = avg_mt1 / avg_st if avg_st > 0 else float("nan")
        r6 = avg_mt6 / avg_st if avg_st > 0 else float("nan")
        rn = avg_mtn / avg_st if avg_st > 0 else float("nan")

        if verbose:
            print(
                f"{nt:>8} | {avg_st:>10.1f} | {avg_mt1:>10.1f} | {avg_mt6:>10.1f} | "
                f"{avg_mtn:>10.1f} | {r1:>7.2f}x | {r6:>7.2f}x | {rn:>7.2f}x"
            )

        f_mt1.close()
        f_mt6.close()
        f_mtn.close()

    if verbose:
        print()
        print(
            f"  MT* = MultiThreadedFitter(n_workers={mt_workers}). "
            "Ratios vs ST: <1.0 means faster than SingleThreaded."
        )
        print()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _compare(nll_a, nll_b, label_a, label_b, verbose, tol=1e-5):
    matches = [abs(a - b) <= tol for a, b in zip(nll_a, nll_b)]
    ok = all(matches)
    if verbose:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label_a} vs {label_b}: "
              f"{sum(matches)}/{len(matches)} iterations match (tol={tol})")
    return ok


def _print_timing_table(rows, verbose):
    if not verbose:
        return
    print()
    hdr = f"{'Fitter':<45} {'avg (ms)':>10} {'total (s)':>12} {'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))
    baseline_avg = None
    for label, times in rows:
        avg = np.mean(times) * 1000
        total = np.sum(times)
        if baseline_avg is None:
            baseline_avg = avg
        speedup = baseline_avg / avg if avg > 0 else float("inf")
        print(f"{label:<45} {avg:>10.2f} {total:>12.4f} {speedup:>9.2f}x")
    print()


# ---------------------------------------------------------------------------
# Pytest entry
# ---------------------------------------------------------------------------

def test_equivalence_and_timing():
    assert run_timing_test(
        n_trials=20,
        n_iterations=20,
        manual_seed=1,
        verbose=False,
        n_repeats=1,
    )


def test_bads_fit_singlethreaded_vs_multithreaded():
    """One BADS ``fit()`` each: ST must match MT(1,6,12); timing table optional."""
    assert run_bads_fit_equivalence_test(
        n_trials=20,
        manual_seed=1,
        feature_drop=0.0,
        n_repeats=1,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SingleThreaded vs MultiThreaded: equivalence and timing"
    )
    parser.add_argument("--data-folder", type=str, default=str(_TESTS_DIR / "data"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--n-repeats", type=int, default=1,
        help="IBS repeats per trial for phase 1–2 (default: 1)",
    )
    parser.add_argument(
        "--n-repeats-scan",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="If set, run ST vs MT timing table for each n_repeats "
        "(e.g. --n-repeats-scan 1 5 10 25 50). Uses --n-iterations-per-scan.",
    )
    parser.add_argument(
        "--n-iterations-per-scan",
        type=int,
        default=5,
        help="Iterations per (n_repeats) cell in --n-repeats-scan (default: 5)",
    )
    parser.add_argument(
        "--trials-scan",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Run ST vs MT1/MT6/MT* by dataset size. "
        "Use with explicit sizes, or pass no values for default 128 256 512. "
        "Rows are tiled if CSV is shorter. Uses --n-repeats-for-trials-scan.",
    )
    parser.add_argument(
        "--bads-mini",
        action="store_true",
        help="Run one short BADS fit per fitter; assert ST == MT(1,6,12) and print timings.",
    )
    parser.add_argument(
        "--practical-bench",
        action="store_true",
        help="Shortcut: --trials-scan 128 256 512 --n-repeats-for-trials-scan 10 "
        "and --skip-phase1 (equivalence + phase-2 timing skipped).",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip equivalence + phase-2 timing; only run optional scans.",
    )
    parser.add_argument(
        "--n-repeats-for-trials-scan",
        type=int,
        default=10,
        help="IBS n_repeats per trial in --trials-scan (default: 10)",
    )
    parser.add_argument(
        "--n-iterations-trials-scan",
        type=int,
        default=5,
        help="Iterations per row in --trials-scan (default: 5)",
    )

    args = parser.parse_args()

    if args.practical_bench:
        args.skip_phase1 = True
        if args.trials_scan is None:
            args.trials_scan = [128, 256, 512]
        elif len(args.trials_scan) == 0:
            args.trials_scan = [128, 256, 512]

    # --trials-scan with no numbers yields []; treat as default grid
    if args.trials_scan is not None and len(args.trials_scan) == 0:
        args.trials_scan = [128, 256, 512]

    if args.bads_mini:
        success = run_bads_fit_equivalence_test(
            data_folder=args.data_folder,
            n_trials=args.n_trials,
            manual_seed=args.seed,
            feature_drop=0.0,
            n_repeats=args.n_repeats,
            verbose=True,
        )
    elif not args.skip_phase1:
        success = run_timing_test(
            data_folder=args.data_folder,
            n_trials=args.n_trials,
            n_iterations=args.n_iterations,
            manual_seed=args.seed,
            verbose=True,
            n_repeats=args.n_repeats,
        )
    else:
        success = True

    if args.n_repeats_scan:
        benchmark_st_vs_mt_n_repeats(
            data_folder=args.data_folder,
            n_trials=args.n_trials,
            n_iterations=args.n_iterations_per_scan,
            manual_seed=args.seed,
            n_repeats_list=tuple(args.n_repeats_scan),
            verbose=True,
        )

    if args.trials_scan is not None:
        benchmark_st_vs_mt_trials(
            data_folder=args.data_folder,
            n_trials_list=tuple(args.trials_scan),
            n_iterations=args.n_iterations_trials_scan,
            manual_seed=args.seed,
            n_repeats=args.n_repeats_for_trials_scan,
            verbose=True,
        )

    sys.exit(0 if success else 1)
