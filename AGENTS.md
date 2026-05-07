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

### Adding New Models
- Subclass `TreeSearch` in `tree_search.py`.
- Override `set_params` to define how BADS parameters are mapped to the heuristic control vector and feature weights.

## Testing Standards
- **Functional Equivalence**: Any change to the fitting logic MUST maintain equivalence. Run `python model_fitting/tests/search_test.py`.
- **Heuristic Properties**: Verify that weights and scales are applied correctly using `python model_fitting/tests/feature_test.py`.
- **Performance**: Monitor execution time using `python model_fitting/tests/timing_test.py`.

## Directory Map
- `/`: C++ source and build configuration.
- `/model_fitting/`: Core Python logic for modeling and optimization.
- `/model_fitting/tests/`: Comprehensive test suite for verification and benchmarking.
- `/docs/`: API documentation and design notes.
