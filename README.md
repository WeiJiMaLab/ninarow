# N-in-a-Row Game Engine and Heuristic Model Fitting

This repository provides a high-performance C++ implementation of N-in-a-Row games (like Connect Four) with Python bindings, designed for cognitive modeling and AI research. The system includes game engines, search algorithms, heuristic evaluation functions, and tools for fitting computational models to human behavioral data.

## Authorial Credit

This is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Please consider him the author of this code for citation purposes.

This repository was written for compatibility with Python 3.10+ by Tyler Seip and is actively maintained by the members of Wei Ji Ma Lab.

## Overview

The repository consists of three main components:

1. **C++ Game Engine**: High-performance game logic, board representation, and search algorithms
2. **Python Bindings**: SWIG-generated Python interface for the C++ components
3. **Model Fitting Framework**: Tools for fitting computational models to human behavioral data

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

**Fit a model to behavioral data:**
```bash
cd model_fitting
python model_fit.py <path_to_game_csv>
```

**Interactive board exploration:**
```bash
cd model_fitting
python board_explorer.py
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

| File | Purpose |
|------|---------|
| `model_fitting/model_fit.py` | Main model fitting pipeline |
| `model_fitting/tree_search.py` | Tree search implementations in Python |
| `model_fitting/board_explorer.py` | Interactive GUI for exploring game positions |
| `model_fitting/examples.py` | Example usage and demonstrations |
| `model_fitting/parsers.py` | Data parsing utilities |
| `model_fitting/utils.py` | General utility functions |
| `model_fitting/ninarow_plotting.py` | Visualization tools |
| `model_fitting/ninarow_utilities.py` | Game-specific utilities |
| `model_fitting/feature_utilities.py` | Feature extraction and manipulation |
| `model_fitting/calculate_summary_statistics.py` | Statistical analysis tools |
| `model_fitting/calculate_tree_statistics.py` | Tree-based statistical analysis |

### Unit Tests

All C++ components include comprehensive unit tests:

| File | Tests |
|------|-------|
| `*_ut.cpp` | Unit tests for corresponding header files |

### Documentation and Examples

| Directory/File | Purpose |
|----------------|---------|
| `docs/` | Auto-generated Doxygen documentation |
| `demos/` | Jupyter notebooks with examples and tutorials |
| `model_fitting/example_inputs/` | Sample data files for testing |
| `model_fitting/heuristic_quality_inputs/` | Benchmark data for heuristic evaluation |

### Development Tools

| File | Purpose |
|------|---------|
| `utils/precommit.sh` | Pre-commit hooks for code quality |
| `utils/run-clang-format.py` | Code formatting utility |
| `Doxyfile` | Doxygen documentation configuration |

## Key Features

### Game Engine
- **High Performance**: Optimized C++ implementation with efficient board representation
- **Flexible**: Supports various N-in-a-Row game variants (Connect Four, etc.)
- **Search Algorithms**: Multiple search strategies including best-first search
- **Heuristic Evaluation**: Sophisticated position evaluation using multiple features

### Model Fitting
- **Bayesian Optimization**: Uses PyBADS for efficient parameter optimization
- **Parallel Processing**: Multi-threaded model fitting for large datasets
- **Rich Features**: Extensive set of heuristic features for position evaluation
- **Visualization**: Interactive tools for exploring model behavior

### Python Integration
- **SWIG Bindings**: Seamless integration between C++ and Python
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Memory Efficient**: Minimal overhead for large-scale computations

## Data Format

The model fitting expects CSV files with game data containing:
- Game positions
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

```bash
cd build
./tests
```

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

## Troubleshooting

### Common Issues

**SWIG Library Errors**: If you see "Unable to find 'swig.swg'" errors, clean your build directory:
```bash
rm -rf build && mkdir build && cd build && cmake ..
```

**Python Package Installation**: On macOS with Homebrew Python, you may need to use a virtual environment or `--break-system-packages` flag.

**CMake Version**: Ensure you have CMake 3.15 or later installed.

## Documentation

- **API Documentation**: [https://weijimalab.github.io/ninarow/](https://weijimalab.github.io/ninarow/)
- **Examples**: See Jupyter notebooks in `demos/`
- **Legacy Documentation**: `legacy/README.md` contains additional setup information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run pre-commit checks: `utils/precommit.sh`
4. Submit a pull request

## License

Please refer to the original repository for licensing information.
