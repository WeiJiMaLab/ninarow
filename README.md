# ninarow

This project simulates the game **N-in-a-Row** (like Connect Four, but on a
4x9 board where you're trying to get 4 in a row) and fits a computational
model of how a person or animal actually plays it.

Concretely, there are two parts:

1. A fast **game engine** written in C++. It knows the rules, can search
   through possible future moves, and can score how good a position looks
   using a "heuristic" (a scoring function with adjustable weights, similar
   in spirit to how a chess engine values a position).
2. A **Python fitting pipeline** that takes real move data (a person's or
   animal's actual choices in the game) and searches for the heuristic
   weights that best explain those choices. This is the part most people
   working in this repo will actually use day to day.

The C++ engine is exposed to Python through SWIG (a tool that auto-generates
a Python wrapper around C++ code), so from Python it just looks like a normal
importable module (`fourbynine`).

## Where things live

| Folder | What's in it |
|---|---|
| `cpp/` | The C++ game engine — board rules, search algorithm, heuristic scoring. |
| `tests/cpp/` | Tests for the C++ code. |
| `tests/python/` | Tests for the Python fitting code. |
| `model_fitting/` | The Python fitting pipeline. **This is where you'll spend most of your time.** Has its own README with a full walkthrough — see below. |
| `demos/` | A small example script. |
| `data/` | Not tracked by git. Where you put your own move data to fit models to. |
| `docs/` | Auto-generated reference docs for the C++ code (built with Doxygen). |

If you open `cpp/`, here's roughly what each file does:

| File | What it's for |
|---|---|
| `ninarow_board.h` | Represents the board and checks whether a move is legal / a win. |
| `ninarow_move.h` | Represents a single move. |
| `ninarow_pattern.h` | Represents a shape/pattern of squares on the board (used to define features, see below). |
| `ninarow_heuristic.h` | Holds the weights the model uses to score a position. |
| `ninarow_heuristic_feature.h` | One scoring ingredient — e.g. "does this pattern of 3-in-a-row exist here." |
| `ninarow_vectorized_feature_evaluator.h` | The fast bit-twiddling code that actually checks for those patterns. |
| `game_tree_node.h`, `bfs_node.h` | Represent one position in the "tree" of possible future moves the search explores. |
| `ninarow_bfs.h`, `searches.h` | The actual search algorithm that looks ahead and picks a move. |
| `player.h` | Which player (Black/White) is moving. |
| `fourbynine_features.h` | The specific feature definitions used for the 4x9 board. |
| `fourbynine.i` | The SWIG file that turns all of the above into a Python module. |
| `CMakeLists.txt` | Build instructions for the C++ code. |

## Building the C++ engine

You need Python 3.10+, CMake, SWIG, and Boost installed. Then, from a fresh
virtual environment:

```bash
python3 -m venv n-in-a-row
source n-in-a-row/bin/activate
chmod +x autobuild.sh
./autobuild.sh
```

`autobuild.sh` figures out what machine you're on (Mac, an NYU cluster, a
generic SLURM cluster, Windows) and handles the rest: installing
CMake/SWIG/Boost if needed, compiling the C++ code, checking that
`import fourbynine` works in Python, and installing the Python packages
listed in `model_fitting/requirements.txt`.

One thing to know: the compiled Python module is tied to whichever Python
interpreter built it. If you later run scripts with a *different* Python
(a different conda env, a different `venv`), importing `fourbynine` will
crash instead of just failing cleanly. If that happens, re-run
`autobuild.sh` with the Python you actually intend to use.

## Fitting models to data

This is covered in detail in **[`model_fitting/README.md`](model_fitting/README.md)**,
which walks through: what a "heuristic" and a "feature" are in plain terms,
how to fit a model to one dataset, and how to run a full multi-attempt fit
across many participants (either on your own machine or on a SLURM cluster).
The short version: `model_fitting/scripts/fit_all.py` is the tool for doing
that (multiple random restarts, across multiple participants, at scale).

## Running the tests

C++ tests:

```bash
cd build
./tests
```

Python tests:

```bash
cd model_fitting
python -m pytest ../tests/python/
```

## If something breaks

**"Unable to find swig.swg" or other SWIG errors** — your build directory is
stale. Wipe it and reconfigure:

```bash
rm -rf build && mkdir build && cd build && cmake ../cpp
```

**`import fourbynine` crashes or segfaults** — you're almost certainly using
a different Python interpreter than the one that built the extension. Make
sure you're running from `model_fitting/` (or have added it to your path)
with the same Python `autobuild.sh` used:

```python
import sys
sys.path.insert(0, '/path/to/ninarow/model_fitting')
```

## Where this code comes from

This project is a rewrite of Bas van Opheusden's [Four-in-a-row
implementation](https://github.com/basvanopheusden/fourinarow). If you're
citing this code, please credit him as the original author.

It was adapted for modern Python (3.10+) by Tyler Seip, and is maintained by
members of the Wei Ji Ma Lab, including Jordan Lei.

There's no formal license file in this repository yet — if you have
questions about reuse, ask the Wei Ji Ma Lab or refer to the original
repository above. There's also no formal contributing guide; if you want to
propose a change, open a pull request against
[WeiJiMaLab/ninarow](https://github.com/WeiJiMaLab/ninarow) and someone will
follow up.
