# Running ninarow model fitting on the NYU Torch HPC cluster

Practical, cluster-specific notes for building and fitting on **Torch**
(`login.torch.hpc.nyu.edu`, Slurm, account `torch_pr_362_general`). Written
while getting Jordan's refactored `model_fitting/scripts/fit_all.py` pipeline
running for a 158-participant refit. Read this before building or submitting.

> TL;DR: build the SWIG `.so` against an **old glibc sysroot (2.17)** or it
> won't import on compute nodes; run all **heavy conda solves on a compute node**
> (`srun`), never the login node; the conda env needs `numba` + `pyyaml` (missing
> from `requirements.txt`) and `numpy<2.3` (scipy 1.14.1 pin).

---

## 1. Login & connection

- SSH: `ssh jo2229@login.torch.hpc.nyu.edu` (MFA / Microsoft login).
- The hostname is a **pool of ~8 login nodes behind one DNS name**, each with its
  own host key — so `ssh` will intermittently warn "REMOTE HOST IDENTIFICATION HAS
  CHANGED". This is expected; add all the pool's host keys to `known_hosts` once.
- For automation, use SSH **connection multiplexing** (ControlMaster) so you
  authenticate (MFA) once and reuse the socket:
  ```
  # ~/.ssh/config
  Host torch
    HostName login.torch.hpc.nyu.edu
    User jo2229
    PreferredAuthentications keyboard-interactive,password
    PubkeyAuthentication no          # avoids "Too many authentication failures"
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 8h
  ```
  (Open the master once interactively: `ssh -fN torch`, complete MFA.)

## 2. The login node vs compute node split (IMPORTANT)

| | Login node | Compute node (`cs*`) |
|---|---|---|
| glibc | **2.39** | **2.34** |
| Heavy conda solves | **killed** (cgroup limit) | OK (with `--mem`) |
| Where jobs run | no | **yes** |

Two consequences that cost real debugging time:

- **Never run big `conda install` / revision rollbacks on the login node** — they
  get `Killed` mid-solve even with free RAM (per-process cgroup cap). Run them on a
  compute node: `srun --account=torch_pr_362_general --time=00:30:00 --mem=24G --cpus-per-task=4 bash -lc '...'`.
- **Build the `.so` for the compute-node glibc, not the login node's** (see §4).

## 3. Conda environment (`env`)

- Base: `source /scratch/jo2229/miniforge3/etc/profile.d/conda.sh` (NOT
  `/scratch/$USER/conda/miniforge3/...` — `autobuild.sh`'s Torch branch has that
  path wrong).
- Project env: `conda activate env` (`/scratch/jo2229/.conda/envs/env`, Python 3.12).
- **Version pins that matter:**
  - `numpy < 2.3` — scipy 1.14.1 (pinned in requirements) refuses numpy ≥ 2.3.
    A stray `conda install` can bump numpy to 2.4.x and silently break scipy.
  - `pandas < 2.3` (2.2.x is known-good; pandas 3.0 got pulled in accidentally once).
- **Deps missing from `model_fitting/requirements.txt`** (install manually):
  - `numba` — imported by `tree_search_fitter.py`; install via **pip**
    (`python -m pip install numba`) so it can't drag conda numpy/pandas up.
    (Installing numba via `conda` migrated python/swig/boost defaults→conda-forge and
    broke the env — avoid.)
  - `pyyaml` — `tree_search.py` / `tree_search_fitter.py` read `config.yaml`.
- Build tools live in `env`: `swig` 4.3.1, `cmake` 4.x, boost headers, and
  conda-forge `c-compiler`/`cxx-compiler` (gcc 14.3). If compilers are absent:
  `conda install -n env -c conda-forge c-compiler cxx-compiler` (on a compute node).

## 4. Building the C++/SWIG extension (`_swig_fourbynine.so`)

CMake root is **`cpp/`** (not the repo root). Eigen + googletest are pulled via
`FetchContent` at configure time (needs internet; the configure step is slow,
~4 min, because it git-clones them).

**Critical — glibc sysroot.** The conda compiler install pulls
`sysroot_linux-64` matching the *login* node (glibc 2.39). A `.so` built that way
imports on the login node but fails on compute nodes with:
```
ImportError: /lib64/libc.so.6: version `GLIBC_2.38' not found (required by _swig_fourbynine.so)
```
Fix: install an **old sysroot** and build against it, so the binary's glibc floor
is ≤ the compute nodes' 2.34:
```
conda install -n env -c conda-forge -y "sysroot_linux-64=2.17"
```

**Build recipe** (run on a compute node to verify against the real glibc):
```bash
source /scratch/jo2229/miniforge3/etc/profile.d/conda.sh
conda activate env
export BOOST_ROOT=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot
cd /scratch/jo2229/ninarow_refit
rm -rf build && mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) -DSWIG_EXECUTABLE=$(which swig) -Dgtest_discover_tests=OFF ../cpp
cmake --build . --config Release -j4
# .so lands in ../model_fitting/_swig_fourbynine.so
python ../tests/python/check_installation.py       # sanity import
```
Verify the glibc floor: `objdump -T model_fitting/_swig_fourbynine.so | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail`.

The `.so` is linked against `env`'s libpython — **always run fitting with that same
Python** (a different interpreter segfaults).

## 5. The fitting pipeline (Jordan's refactor)

`model_fitting/scripts/`:
- `fit_all.py data_dir n_splits [--n-starts 5] [--n-repeats 40] [--yes]` — interactive
  driver. Grid = participants × folds; each grid point = `n_starts` BADS multistarts
  → a "reduce" that re-evaluates all starts at high IBS repeats and argmin-picks a winner.
  - **Multistarts** = `--n-starts` (default 5). **Folds** = `n_splits` (positional).
  - Parallel path submits, per grid point: a SLURM array of `n_starts` fit tasks
    (`fit_one_start.py`) + a dependent `--dependency=afterok` reduce job (`consolidate.py`).
  - `--yes` skips only the final confirm; **mode / account / cores / time are still
    interactive `input()`** — for headless runs, drive `fit_one_start.py` +
    `consolidate.py` directly from your own sbatch script.
- `fit_one_start.py data_dir n_splits held_out_index --start S --n-repeats N --n-workers W`
  — fully arg-driven (no prompts); writes `starts/<held_out>.<start>.json`.
- `consolidate.py data_dir n_splits held_out_index n_starts --n-workers W`
  — winner selection; writes `results/<held_out_index>.json`.

**Parameters** (`param_names`, matches the old code's order exactly):
`pruning_threshold, stopping_prob, feature_drop, lapse_rate, opp_scale, center_weight, 2IAR_CON, 2IAR_DIS, 3IAR, 4IAR`.

**`feature_drop` caveat:** the production path currently *fits* `feature_drop`
(`default_model_factory(exclude_feature_drop=False)` — no CLI flag wired). The
`exclude_feature_drop=True` machinery exists (drops it, re-inserts 0.0). For
production it should be excluded — pending an upstream flag from Jordan.

## 6. Data format

Each participant dir must contain **exactly `n_splits` fold CSVs** named
`0.csv .. (n_splits-1).csv`, comma-separated **with header**, required columns
`black, white, move, color` (extra cols allowed). One held-out fold = test, the
rest = train.

The lab's original real data (`/scratch/jo2229/splits/<p>/`) is the OLD format:
TAB-separated, no header, columns `black white color move time group participant`,
pre-split into 5 folds (`1.csv..5.csv`) + full `data.csv`. Convert with
`make_3fold_data.py` (re-splits each participant's positions into 3 folds; matches
the old `generate_splits`: random per-position round-robin, no game grouping).

## 7. Slurm quick reference

- Account: `--account=torch_pr_362_general`. Default partition allocates CPU jobs.
- Interactive/one-off compute: `srun --account=torch_pr_362_general --time=... --mem=... --cpus-per-task=... <cmd>`.
- `srun` stdout doesn't always stream back over a piped SSH session — redirect to a
  file in `/scratch` and read that.

---

## Appendix: build/env issues hit (and fixes) — 2026-07-20

| Symptom | Cause | Fix |
|---|---|---|
| `conda install` `Killed` on login | login-node cgroup cap | run on compute node via `srun` |
| `ImportError: GLIBC_2.38 not found` on compute | `.so` built vs sysroot 2.39 | `sysroot_linux-64=2.17`, rebuild |
| `ModuleNotFoundError: numba` | missing from requirements | `pip install numba` |
| `No module named 'yaml'` | missing from requirements | `conda install -n env pyyaml` |
| scipy warns numpy 2.4.6 unsupported | conda numba pulled numpy up | pin `numpy<2.3 pandas<2.3` |
| CMake "no CMakeLists.txt" | root moved to `cpp/` | point cmake at `../cpp` |
| autobuild sources wrong conda path | Torch branch bug in `autobuild.sh` | source `/scratch/jo2229/miniforge3/...` |
