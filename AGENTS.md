# Agentic Development Guide: N-in-a-Row

This document provides a high-level architectural overview and development guidelines for AI agents working on this repository.

## Project Vision
N-in-a-Row is a research-grade platform for cognitive modeling. It bridges high-performance C++ game logic with flexible Python-based model fitting. The primary goal is to provide a robust environment for fitting computational models (like BFS with heuristic evaluation) to human behavioral data.

## Core Architecture

### 1. C++ Core (`fourbynine`)
- **SWIG Bindings**: The project uses SWIG to expose C++ classes to Python. The compiled module is typically accessed as `fourbynine`.
- **Key Classes**:
    - `fourbynine_board`: Represents the game state.
    - `fourbynine_heuristic`: Evaluates board positions.
    - `NInARowBestFirstSearch`: The primary search engine.

### 2. Modular Heuristic Framework (`model_fitting/`)
Instead of hardcoding features in C++, we use a modular system:
- **`tree_search.py`**: Defines model classes (`TreeSearch`, `MyopicTreeSearch`). These classes hold the template definitions and manage the mapping between optimization parameters and heuristic construction.
- **`feature_generator.py`**: Contains the logic for tiling templates across the board and converting them into C++ compatible features.
- **`tree_search_fitter.py`**: Provides the execution engines for optimization.
    - `SingleThreadedFitter`: Sequential processing, ideal for debugging.
    - `MultiThreadedFitter`: Parallelized processing using `multiprocessing.Pool`.
- **`initial_values`**: Models now support an `initial_values` dictionary in their constructor to explicitly set starting parameters for specific feature groups.

## Common Development Tasks

### Modifying Heuristics
1. Update `DEFAULT_TEMPLATES` in `tree_search.py` or provide custom templates to the `TreeSearch` constructor.
2. If you change the number of parameters, ensure `compile_parameters()` is called to update the optimization bounds.

### Debugging Fitting Issues
- Use `SingleThreadedFitter` to avoid the complexity of multiprocessing during debugging.
- Check `model_fitting/tests/feature_test.py` to verify heuristic evaluation properties (e.g., `opp_scale` intuition) and `model_fitting/tests/search_test.py` for end-to-end consistency.
- **BADS tolerance smoke:** `python -m logistic_smoke` only when needed; one pytest in `test_bads_logistic_tolerance.py`. Production IBS: `tol_fun` tight, stop on `tol_mesh`.

### Adding New Models
- Subclass `TreeSearch` in `tree_search.py`.
- Override `set_params` to define how BADS parameters are mapped to the heuristic control vector and feature weights.

## Testing Standards
- **Functional Equivalence**: Any change to the fitting logic MUST maintain equivalence. Run `python model_fitting/tests/search_test.py`.
- **Heuristic Properties**: Verify that weights and scales are applied correctly using `python model_fitting/tests/feature_test.py`.
- **Performance**: Monitor execution time using `python model_fitting/tests/timing_test.py`.

### Smoke tests (agents)
A **smoke test must finish in ~1–2 minutes**. If it runs longer, stop — you are not running a smoke test.

- **Do not** run `pytest tests/test_bads_logistic_tolerance.py` in bulk agent loops, multi-fold CV grids, or `--benchmark` unless the user explicitly asks.
- **Do not** chain smoke + full pytest + benchmark in one session (memory and wall time).
- Default: `python -m logistic_smoke` — one tiny split, five settings, `SMOKE_MAX_FUN_EVALS=60` (sanity only; production fits use 2000 evals).
- Report existing smoke output when present; do not re-run to “confirm”.
- Production fitter BADS: `tol_mesh=1e-3`, `tol_fun=1e-7`. Fair logistic baseline (for later): **scipy** `ftol=gtol=1e-6` (prefer over sklearn on this toy setup).

## Directory Map
- `/`: C++ source and build configuration.
- `/model_fitting/`: Core Python logic for modeling and optimization.
- `/model_fitting/tests/`: Comprehensive test suite for verification and benchmarking.
- `/docs/`: API documentation and design notes.

---

## Change Log

### 2026-05-25: BADS tolerance smoke test (logistic proxy)
- Added `model_fitting/logistic_smoke.py` and `tests/test_bads_logistic_tolerance.py` (one fast pytest).
- Default CLI: one tiny split, capped BADS evals; `--benchmark` is slow and not for agents.
- Documented production tols (`tol_mesh=1e-3`, `tol_fun=1e-7`) vs scipy `ftol=gtol≈1e-6` on smooth logistic NLL. See `model_fitting/NOTES_bads_tolerance.md`.

### 2026-05-08: Fitter Performance & Numba Acceleration
**Author: Antigravity AI**

#### 1. Core Performance Refactor (tree_search_fitter.py)
- **Numba-Accelerated IBSTracker:** Refactored the IBS accumulation logic into a `jitclass`. This eliminated Python list overhead in the core loop, drastically reducing evaluation latency.
- **Empirical Noise Reporting:** Updated `optimize` to calculate and print `NLL ± SD`, providing real-time visibility into objective function stability.

#### 2. Optimizer Stabilization
- **Spatial Termination Strategy:** Implemented `tol_mesh: 1e-3` as the primary stopping criterion in the default fitters.
- **Two-Stage Warm Start:** Refactored the fitting entry points to support an optional Stage 1 (global search, low repeats) and Stage 2 (local refinement, high repeats) workflow.
- **Initialization Optimization:** Reduced default `fun_eval_start` to `20` to minimize Sobol-sequence overhead.
