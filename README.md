# N-in-a-Row Game Engine and Heuristic Model Fitting

This repository provides a high-performance C++ implementation of N-in-a-Row games (like Four-in-a-Row) with Python bindings, designed for cognitive modeling and AI research. The system includes game engines, search algorithms, heuristic evaluation functions, and tools for fitting computational models to human behavioral data.

## Table of Contents
- [Authorial Credit](#authorial-credit)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
  - [Core C++ Components](#core-c-components)
  - [Build System](#build-system)
  - [Python Model Fitting Framework](#python-model-fitting-framework)
  - [Testing Framework](#testing-framework)
  - [Documentation and Examples](#documentation-and-examples)
- [Model Fitting: Dual Implementation Approach](#model-fitting-dual-implementation-approach)
  - [model_fit.py: Original Implementation](#modelfitpy-original-implementation)
  - [tree_search_fitter.py: Specialized Fitters](#tree_search_fitterpy-specialized-fitters)
  - [When to Use Which](#when-to-use-which)
- [Modular Heuristic Design](#modular-heuristic-design)
- [Key Features](#key-features)
- [Data Format](#data-format)
- [Development](#development)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Authorial Credit

This is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Please consider him the author of this code for citation purposes.

This repository was written for compatibility with Python 3.10+ by Tyler Seip and is actively maintained by the members of Wei Ji Ma Lab.

## Overview

The repository consists of three main components:

1. **C++ Game Engine**: High-performance game logic, board representation, and search algorithms
2. **Python Bindings**: SWIG-generated Python interface for the C++ components
3. **Model Fitting Framework**: Tools for fitting computational models to human behavioral data

The model fitting framework provides two complementary implementations:
- **`tree_search.py`**: Modern modular model definitions using template-based heuristic construction
- **`tree_search_fitter.py`**: Specialized fitting engines (Single and Multi-threaded) for modular models

Both implementations produce identical results but offer different levels of flexibility for feature definition and heuristic construction.

## Quick Start

### Prerequisites

- **Python 3.10+** 
- **CMake** (for building C++ components)
- **SWIG** (for generating Python bindings)
- **Boost** (C++ libraries)

### Installation

**We strongly recommend using a virtual environment:**
```bash
python3 -m venv n-in-a-row
source n-in-a-row/bin/activate  # On Windows: n-in-a-row\Scripts\activate
```

**Build the project:**
```bash
chmod +x autobuild.sh
./autobuild.sh
```

The script will:
- Install system dependencies (CMake, SWIG, Boost)
- Build the C++ components
- Generate Python bindings
- Run unit tests
- Install Python packages

### Basic Usage

**Fit a model using the modular implementation (recommended):**
```bash
cd model_fitting
python tree_search.py
```

**Fit a model using the original implementation:**
```bash
cd model_fitting
python model_fit.py <path_to_game_csv>
```

Note that the parameter bounds in the original implementation are as follows: 
- pruning threshold
- stopping probability
- feature drop rate
- lapse rate
- active / passive scaling
- center feature
- connected 2IAR
- disconnected 2IAR
- 3IAR
- 4IAR

Note that you may want to change the parameter fitting bounds
for ease of fitting, consider increasing the lower bound
on the second parameter (stopping probability; 0.001 --> 0.01)
and the plausible lower bound of the fourth parameter (lapse rate; 0.001 --> 0.05)


**Interactive board exploration:**
```bash
cd model_fitting
python board_explorer.py
```

**Run tests:**
```bash
cd model_fitting
python tests/feature_test.py
python tests/search_test.py
```

## Repository Structure

### Core C++ Components

| File | Purpose |
|------|---------|
| `ninarow_board.h` | Game board representation and move validation |
| `ninarow_move.h` | Move representation and utilities |
| `ninarow_pattern.h` | Pattern matching for game positions |
| `ninarow_heuristic.h` | Heuristic evaluation functions |
| `ninarow_heuristic_feature.h` | Individual heuristic features |
| `ninarow_vectorized_feature_evaluator.h` | Optimized feature evaluation |
| `game_tree_node.h` | Game tree node representation |
| `bfs_node.h` | Best-first search node implementation |
| `ninarow_bfs.h` | Best-first search algorithm |
| `searches.h` | Various search algorithm implementations |
| `player.h` | Player representation and utilities |

### Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | CMake build configuration |
| `autobuild.sh` | Automated build script for all platforms |
| `fourbynine.i` | SWIG interface definition |
| `fourbynine_features.h` | Feature definitions for 4×9 Connect Four variant |

### Python Model Fitting Framework

#### Core Model Fitting Modules

| File | Purpose |
|------|---------|
| `model_fit.py` | Original model fitting pipeline using parameter vector approach |
| `tree_search.py` | Modular model definitions (TreeSearch, Myopic, etc.) |
| `tree_search_fitter.py` | Fitting engines (SingleThreadedFitter, MultiThreadedFitter) |
| `feature_generator.py` | Modular heuristic construction from templates |
| `ninarow_utilities.py` | Game-specific utilities and parameter conversion |

#### Supporting Modules

| File | Purpose |
|------|---------|
| `board_explorer.py` | Interactive GUI for exploring game positions |
| `examples.py` | Example usage and demonstrations |
| `parsers.py` | Data parsing utilities |
| `utils.py` | General utility functions |
| `ninarow_plotting.py` | Visualization tools |
| `feature_utilities.py` | Feature extraction and manipulation |
| `calculate_summary_statistics.py` | Statistical analysis tools |
| `calculate_tree_statistics.py` | Tree-based statistical analysis |

### Testing Framework

| Directory/File | Purpose |
|----------------|---------|
| `model_fitting/tests/feature_test.py` | Heuristic evaluation properties (weights, `opp_scale` intuition) |
| `model_fitting/tests/search_test.py` | System-level consistency and search determinism |
| `model_fitting/tests/timing_test.py` | Performance benchmarks and timing |
| `model_fitting/logistic_smoke.py` | Fast PyBADS vs scipy tolerance check on tiny logistic NLL (not TreeSearch/IBS) |
| `model_fitting/tests/test_bads_logistic_tolerance.py` | Single-split pytest sanity check (optional; see smoke notes below) |
| `test_cpp/*_ut.cpp` | C++ unit tests for corresponding header files |

### Documentation and Examples

| Directory/File | Purpose |
|----------------|---------|
| `docs/` | Auto-generated Doxygen documentation |
| `demos/` | Jupyter notebooks with examples and tutorials |
| `model_fitting/example_inputs/` | Sample data files for testing |
| `model_fitting/heuristic_quality_inputs/` | Benchmark data for heuristic evaluation |
| `legacy/` | Legacy code and documentation |

### Development Tools

| File | Purpose |
|------|---------|
| `utils/precommit.sh` | Pre-commit hooks for code quality |
| `utils/run-clang-format.py` | Code formatting utility |
| `Doxyfile` | Doxygen documentation configuration |

## Model Fitting: Dual Implementation Approach

The repository provides two implementations for model fitting that are **functionally equivalent**—they produce identical NLL (negative log-likelihood) values when configured with matching parameters.

### Functional Equivalence

Both implementations use Inverse Binomial Sampling (IBS) to estimate log-likelihoods of model parameters given observed human moves. Despite architectural differences, they produce identical results:

| Aspect | `model_fit.py` | `tree_search_fitter.py` |
|--------|----------------|------------------|
| Heuristic construction | 58-parameter vector, 17 feature groups | Template-based, custom feature groups |
| Trial processing | Interleaved across trials | Sequential per trial |
| Parallelization | Multiprocessing pool | Single or Multi-threaded engines |
| **NLL output** | **Identical** | **Identical** |

The test suite (`tests/search_test.py`) verifies this by running both implementations on the same data with matched parameters and confirming log-likelihood values match to within floating point precision.

### model_fit.py: Original Implementation

**Purpose**: Legacy-compatible implementation optimized for parallel processing.

**Key Characteristics**:
- Uses `bads_parameters_to_model_parameters()` to expand 10 BADS parameters into 58-parameter vector
- Constructs heuristics via `fourbynine_heuristic.create(DoubleVector(params), True)`
- Processes trials in interleaved fashion with early stopping based on cumulative NLL cutoff
- Supports multi-threaded execution via `multiprocessing.Pool`

**Example Usage**:
```python
from model_fit import DefaultModel, ModelFitter

model = DefaultModel()
fitter = ModelFitter(args, model)
params, loglik = fitter.fit_model(moves)
```

### tree_search_fitter.py: Specialized Fitters

**Purpose**: Modern, flexible implementation with explicit template-based heuristic construction and optimized fitting engines.

**Key Characteristics**:
- Accepts modular models (TreeSearch, Myopic) as input
- Builds heuristics from template definitions via `TreeSearch.create_heuristic()`
- Supports both **`SingleThreadedFitter`** (sequential) and **`MultiThreadedFitter`** (parallel)
- Processes each trial sequentially to completion for robust NLL estimation

**Example Usage**:
```python
from tree_search import TreeSearch
from tree_search_fitter import MultiThreadedFitter

# 1. Initialize modular model
model = TreeSearch()

# 2. Select a fitting engine (e.g. 6 workers)
fitter = MultiThreadedFitter(model, n_workers=6)

# 3. Fit to behavioral data
fitted_params, final_LL = fitter.fit(data)
```

### When to Use Which

| Scenario | Recommended |
|----------|-------------|
| New projects | `tree_search.py` |
| Large datasets (parallelization needed) | `model_fit.py` |
| Custom feature development | `tree_search.py` |
| Cross-language verification (Python ↔ Julia) | `tree_search.py` |
| Legacy code compatibility | `model_fit.py` |

## Modular Heuristic Design

The modular design (`tree_search.py` + `feature_generator.py`) allows heuristics to be constructed from templates:

### Templates

Templates define patterns to match on the board. Each template is a list of 0s and 1s representing piece positions:

```python
DEFAULT_TEMPLATES = {
    "4IAR": [[1, 1, 1, 1]],  # Four in a row
    "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], ...],  # Three in a row (connected)
    "2IAR_CON": [[1, 1, 0, 0], ...],  # Two in a row (connected)
    "2IAR_DIS": [[1, 0, 0, 1], ...],  # Two in a row (disconnected)
}
```

### Feature Generation

The `feature_generator.py` module:
- Generates all possible board positions for each template
- Creates bitboard representations (pieces and spaces)
- Groups features by template type
- Constructs heuristics using `create_modular_heuristic()`

### Benefits

1. **Flexibility**: Add new feature types without modifying C++ code
2. **Clarity**: Templates are human-readable pattern definitions
3. **Consistency**: Same templates work across Python and Julia implementations
4. **Maintainability**: Feature definitions are centralized in Python

### Verification

The test suite verifies functional equivalence at multiple levels:

**`tests/feature_test.py`** — Heuristic properties and intuition:
- Modular heuristics produce expected `evaluate()` values based on group weights and `opp_scale`.
- Verifies that `opp_scale = 0.5` correctly weights the opponent's features at half the value of the current player's.
- Verifies that `initial_values` are correctly applied during model construction.

**`tests/search_test.py`** — End-to-end system consistency:
- Runs both `SingleThreadedFitter` and `MultiThreadedFitter` on identical data.
- Verifies search determinism when noise is disabled.

## Key Features

### Game Engine
- **High Performance**: Optimized C++ implementation with efficient board representation
- **Flexible**: Supports various N-in-a-Row game variants (Connect Four, etc.)
- **Search Algorithms**: Multiple search strategies including best-first search
- **Heuristic Evaluation**: Sophisticated position evaluation using multiple features

### Model Fitting
- **Bayesian Optimization**: Uses PyBADS for efficient parameter optimization
- **Parallel Processing**: Multi-threaded model fitting for large datasets
- **Modular Design**: Template-based heuristic construction for flexibility
- **Dual Implementation**: Both legacy and modern approaches available
- **Rich Features**: Extensive set of heuristic features for position evaluation
- **Visualization**: Interactive tools for exploring model behavior

### Python Integration
- **SWIG Bindings**: Seamless integration between C++ and Python
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Memory Efficient**: Minimal overhead for large-scale computations
- **Cross-Language**: Compatible with Julia implementations

## Data Format

The model fitting expects CSV files with game data containing:
- Game positions (bitboard representations)
- Human moves
- Game outcomes
- Timing information (optional)

See `model_fitting/example_inputs/` for sample data formats.

## Development

### Building from Source

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Running Tests

**C++ Unit Tests:**
```bash
cd build
./tests
```

**Python Tests:**
```bash
cd model_fitting
python tests/feature_test.py
python tests/search_test.py
```

**BADS tolerance smoke** (tiny logistic NLL; target ~1–2 minutes — not a full fit benchmark):

```bash
cd model_fitting
python -m logistic_smoke
```

Compares production PyBADS stops (`tol_mesh=1e-3`, `tol_fun=1e-7`) to scipy on synthetic data. Use this to sanity-check optimizer settings before long IBS/TreeSearch jobs. Optional slow grid: `python -m logistic_smoke --benchmark`. See `model_fitting/NOTES_bads_tolerance.md` and `AGENTS.md` (Smoke tests).

Production IBS fitting should stop primarily on **`tol_mesh`**, with **`tol_fun`** tight so noisy NLL plateaus do not terminate early. A fair deterministic logistic baseline (for separate comparison work) is scipy L-BFGS-B with `ftol=gtol≈1e-6`.

### Code Style

Before committing, run:
```bash
cd utils
./precommit.sh
```

## Dependencies

### System Dependencies
- CMake 3.15+
- SWIG 4.0+
- Boost libraries
- C++14 compatible compiler

### Python Dependencies
See `model_fitting/requirements.txt` for the complete list:
- numpy, scipy, matplotlib
- pandas (data handling)
- PyQt6 (GUI components)
- PyBADS (Bayesian optimization)
- tqdm (progress bars)
- UltraDict (shared memory dictionaries)

## Troubleshooting

### Common Issues

**SWIG Library Errors**: If you see "Unable to find 'swig.swg'" errors, clean your build directory:
```bash
rm -rf build && mkdir build && cd build && cmake ..
```

**Python Package Installation**: On macOS with Homebrew Python, you may need to use a virtual environment or `--break-system-packages` flag.

**CMake Version**: Ensure you have CMake 3.15 or later installed.

**Module Import Errors**: If importing `tree_search` fails, ensure you're in the `model_fitting` directory or have added it to your Python path:
```python
import sys
sys.path.insert(0, '/path/to/ninarow/model_fitting')
```

## Documentation

- **API Documentation**: [https://weijimalab.github.io/ninarow/](https://weijimalab.github.io/ninarow/)
- **Examples**: See Jupyter notebooks in `demos/`
- **Legacy Documentation**: `legacy/README.md` contains additional setup information
- **Test Files**: See `test/` directory for usage examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run pre-commit checks: `utils/precommit.sh`
4. Run tests: `python tests/feature_test.py`
5. Submit a pull request

## License

Please refer to the original repository for licensing information.
