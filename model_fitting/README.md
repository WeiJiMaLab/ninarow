# Model Fitting

This is the Python side of the `ninarow` project. It answers the question:
**given a set of moves a person or animal actually made in the game, what
set of "scoring weights" would make a computer model play most like them?**

If you're new to this codebase, read the Concepts section first — the rest
of this README assumes you know what a "heuristic," a "feature," and a
"search" are in this specific context.

## Concepts, explained simply

Think of the model as a simulated player that looks at a board and picks a
move. It does this in two steps:

1. **Scoring**: it looks at a possible future position and assigns it a
   number — how good does this position look for me? This scoring function
   is called the **heuristic**.
2. **Searching**: starting from the current position, it imagines a bunch of
   possible move sequences (like a tree branching out into the future),
   scores where each one leads using the heuristic, and picks a move based
   on that.

The heuristic's score is built from a checklist of pattern-detectors called
**features** — things like "is there an open 3-in-a-row here" or "is there a
4-in-a-row here." Each feature type has a **weight**: how much that pattern
counts toward the total score. A feature "group" is a set of features built
from the same template (e.g. all the possible positions where "4 in a row"
could occur) that share one weight.

So concretely, "fitting a model" means: search over possible combinations of
weights (plus a few other settings, like how deep the search looks ahead)
until you find the combination that makes the simulated player's predicted
moves match the real moves as closely as possible.

In the code:
- `fourbynine_heuristic` (from the compiled C++ code) is the actual scoring
  object — it holds the weights and can score a position.
- `NInARowBestFirstSearch` is the actual search — give it a heuristic and a
  board, and it returns one predicted move. Because there's some randomness
  built in, running the search twice from the same position can give two
  different answers — this matters a lot for fitting, see below.
- **`TreeSearch`** (in `tree_search.py`) is the Python class you'll actually
  use. It's a convenient wrapper: you give it a list of weights, and it
  builds the C++ heuristic + search combination for you, and lets you ask
  "what would this model predict here?" (`.predict(board)`) or "how well
  does this model explain this dataset?" (via the fitters, below).

## What's in this folder

```
model_fitting/
  tree_search.py          The TreeSearch model class (see Concepts above)
  tree_search_fitter.py   The fitting engines (see "How fitting works" below)
  multistart.py           Shared code for running many fit attempts and picking the best one
  scripts/                The tools you actually run for a real fit (see below)
  utils.py                Board/feature math: bitboard patterns, heuristic construction, BADS<->heuristic param conversion
  fourbynine.py / _swig_fourbynine.so   Auto-generated Python binding to the C++ code (don't edit)
  config.yaml             Default starting values and bounds for each weight
  requirements.txt
```

`parameter_recovery/` (the pipeline sanity check described below) lives at
the repo root, as a sibling of `model_fitting/`, not inside this folder.

(An older version of this folder had more files — a GUI board explorer,
some plotting helpers, an older fitting script called `model_fit.py`, etc.
These were removed because nothing used them anymore. If you're looking for
one of them, check the git history. The Python tests for this code live at
`../tests/python/`, not in this folder.)

## The `TreeSearch` model class

`TreeSearch` (in `tree_search.py`) is how you build a model in Python. A few
variants exist for different research questions:

- **`TreeSearch`** — the full model. Has 6 adjustable "control" settings
  (how deep to search, how likely the search is to stop early, etc.) plus
  one weight per feature group.
- **`MyopicTreeSearch`** — same, but always searches to completion (no
  "stop early" setting). Useful for testing whether the "stops early"
  behavior actually matters.
- **`MyopicSelfOnlyTreeSearch`** — same as Myopic, but the model completely
  ignores the opponent's pieces when scoring. Useful for testing whether
  modeling the opponent matters at all.
- **`LesionTreeSearch(group_name, ...)`** — the full model with one feature
  group surgically removed. Useful for asking "how much does this specific
  pattern-type matter to the fit?"

All of these can save/load with `.save(path)` / `TreeSearch.load(path)`.

## How fitting works

Fitting means: try different weight combinations, and score how well each
one explains the real data, until you find the best one.

**The tricky part** is that the model is *stochastic* — ask it to predict a
move from the same position twice, and it might give two different answers.
So you can't just check "did the model's top prediction match the real
move?" — you need a proper likelihood. This code uses a technique called
**Inverse Binomial Sampling (IBS)**: instead of asking the model for a
probability directly, it repeatedly asks the model to predict, and counts
how many tries it took to match the real move. More tries needed = the real
move was less likely under this weight combination. This gives an unbiased
estimate without needing the model to output probabilities directly.

Once you can score how well a weight combination fits, you need an
optimizer to search for a good one. This code uses **BADS** (Bayesian
Adaptive Direct Search, via the `pybads` package), which is built for
exactly this situation — expensive, noisy scoring functions where you can't
just compute a gradient.

Two fitting engines live in `tree_search_fitter.py`:

- **`SingleThreadedFitter`** — does everything one trial at a time. Slower,
  but simple and easy to debug. Good as a reference to check the parallel
  version against.
- **`MultiThreadedFitter`** — the one you'll actually use. Spreads the work
  across multiple CPU cores. A few things worth knowing:
  - It starts with a small number of IBS repeats (cheap, noisy) and
    gradually increases them as the search narrows in (slower, but more
    precise) — this saves a lot of time without losing accuracy where it
    matters.
  - It supports checkpointing: pass `checkpoint_path` and it'll save
    progress as it goes, so a fit that got interrupted (e.g. hit a cluster
    time limit) can resume instead of starting over.
  - **Its `n_workers` setting defaults to using every CPU core on the
    machine.** On a shared cluster login node, that can mean 100+
    processes — see the big warning below before you run anything.

### A note on `model_fit.py`

Earlier versions of this project had a different fitting script called
`model_fit.py`. It's been replaced by `tree_search_fitter.py` and is not
present in this checkout. If you find a copy elsewhere, treat it as
historical — use `scripts/fit_all.py` instead.

## Fitting a dataset: `scripts/fit_all.py`

In practice you want more than a single fit:

- **Multiple random restarts** ("multistart") — the optimizer can get stuck
  in a locally-okay-but-not-great answer, so you fit several times from
  different starting points and keep the best one.
- **Multiple participants/datasets at once** — if you're fitting the same
  kind of model to many people's or animals' data.

`scripts/fit_all.py` is an interactive tool that handles both. Point
it at a folder that contains one subfolder per participant, each holding
its own `0.csv` / `1.csv` / ... :

```
data_dir/
  participant1/0.csv 1.csv 2.csv
  participant2/0.csv 1.csv 2.csv
```

- Each CSV needs columns `black`, `white`, `move`, `color` (numbers
  encoding the board state and what was played, and which side moved).
  Two optional columns, `trial_id` and `n_pieces`, get carried through if
  present but aren't required.

```
python scripts/fit_all.py <data_dir> <n_splits> [--n-starts N] [--n-repeats N] [--n-workers N] [--verbose] [--yes]
```

Try it on the sample data:

```
python scripts/fit_all.py ../data/sample 3 --n-starts 2 --n-repeats 20
```

When you run it, it will:
1. Check the folder exists and has the right number of splits for each
   participant.
2. Tell you exactly how many fitting jobs this adds up to
   (`n_starts x n_splits x n_participants`).
3. Ask whether to run **Sequentially** (on this machine, one job after
   another) or **In Parallel** (submit jobs to a SLURM cluster to run at the
   same time).
4. Ask a few follow-up questions if you chose Parallel (see below), show
   you a summary of what's about to happen, and ask you to confirm before
   doing anything.

(Running this will create `starts/` and `results/` subfolders inside each
participant's directory — see "What the output files look like" below for
what ends up in them.)

### Sequential vs. Parallel

- **Sequential** runs every job on your current machine, one at a time,
  using regular multiprocessing. Good for the sample data, a small
  dataset, or when you're debugging.
- **Parallel** submits jobs to a SLURM cluster: one array job per
  (participant, held-out fold), where each array task runs
  `scripts/fit_one_start.py` for one random restart, followed by a
  `scripts/consolidate.py` job that waits for those to finish and picks the
  best one. You'll be asked for:
  - Your SLURM account.
  - How many CPU cores per job (default: 8).
  - How long to give each job (it suggests a time limit based on how many
    IBS repeats you asked for — more repeats means a slower, more precise
    fit, so it suggests more time. You can always type in your own number
    instead.)

### Important: don't let this run wild on a shared machine

**Never run a fit without an explicit, reasonable core-count limit on a
shared login or compute node.** If you don't tell `MultiThreadedFitter` how
many workers to use, its built-in default is "use however many cores this
machine reports" — on a shared cluster node, that can be 100+ cores, even
though you're not the only person using that machine. Spawning that many
processes will slow down or crash other people's work.

In practice this is already handled for you: `scripts/fit_all.py` picks a
safe default automatically (use the number of cores SLURM actually gave
you, if you're inside a job; otherwise just 6) instead of falling back to
"every core on the machine." But if you're calling the fitting code
directly instead of through this script, always pass an explicit worker
count.

### A couple of SLURM quirks worth knowing

If you're curious why the SLURM-submission code looks the way it does:

- Each job's command (`cd <repo path> && python scripts/...`) is submitted
  inline via `sbatch --wrap`, with the repo path baked in as an absolute
  path rather than relying on the submitted script figuring out its own
  location — under SLURM the working directory at job start isn't
  guaranteed to be where you ran `sbatch` from.
- Log file locations are passed in as full absolute paths, rather than
  relative ones — SLURM resolves relative log paths against the directory
  you submitted from, before the job itself has a chance to set up its own
  working directory, which can point log files somewhere unexpected.

## What the output files look like

**Each fold CSV** (`0.csv`, `1.csv`, ...): one row per move, with columns
`black`, `white`, `move`, `color` (and optionally `trial_id`, `n_pieces`).

**`starts/<fold>.<start>.json`** — the raw result of one fitting attempt
(one random restart, for one held-out fold):
```
{
  "start": 0,
  "x0": { "pruning_threshold": 5.0, "stopping_prob": 0.45, ... },   // where this attempt started
  "params": { "pruning_threshold": 3.2, "stopping_prob": 0.51, ... }, // where it ended up
  "train_nll_raw": 812.4,   // how well it fit the training data (lower = better)
  "status": "ok"            // means the file was written completely and safely
}
```
(`x0` and `params` are dictionaries mapping each parameter's name to its
value, so you can read them without needing to remember what order the
numbers come in.)

**`results/<fold>.json`** — the *best* attempt for one held-out fold, after
double-checking which one actually won (see below):
```
{
  "held_out_index": 0,
  "winning_start": 2,
  "n_starts": 5,
  "params": { ... },
  "x0": { ... },
  "train_nll_raw": 812.4,
  "train_nll_reeval": 809.1,      // re-checked score used to pick the winner
  "train_nll": 809.1,             // final training score
  "test_nll_per_trial": [...],    // held-out score, one number per test move
  "test_nll": 244.7               // final held-out score (sum of the above)
}
```

Why "re-check which one won" is a separate step: each individual fitting
attempt's score is a little noisy (remember, IBS is a randomized estimate).
If you just picked whichever attempt reported the best score, you'd
sometimes pick one that got lucky rather than one that's actually best.  So
after all the attempts finish, every one of them gets re-scored more
carefully (more IBS repeats, averaged over several tries), and only then is
the winner picked. The held-out score is then computed just once, using
only the winner — so that final number isn't biased by the "pick the
luckiest one" problem either.

## `parameter_recovery/`

Lives at the repo root (`../parameter_recovery/` from here), as a sibling of
`model_fitting/`, not inside it.

A sanity check for the whole pipeline: pick a known set of weights, use the
model to generate fake data as if a "true" player with those weights
existed, then refit that fake data through the *same* multistart hot path
`scripts/fit_all.py` uses (`fit_one_start` per random restart, then
`select_winner`'s high-repeat re-evaluation to pick the best one) and see if
it recovers weights close to the ones you started with. If it can't recover
its own known ground truth, something's wrong with the fitting process
itself (not with real data).

- `fast_recover.py` — builds N synthetic "participants" from real board
  positions in `data/sample`, each with its own randomly-drawn ground-truth
  weights, and refits them. Deliberately runs with a smaller multistart
  budget than production (fewer restarts/repeats, and a capped BADS
  evaluation budget) so a full sanity-check run takes minutes, not the
  ~1-day-per-restart production budget — this checks a recoverability
  *lower bound*, not full convergence. Pass `--exclude-feature-drop` to
  freeze `feature_drop` out of the fitted parameter vector entirely
  (mirrors the `monkey_4iar` production regime) instead of merely pinning
  it to a narrow range. Writes `data/recovery/participant<i>/0.csv` (fake
  trial data, same schema as `data/sample`) and `recovery.json`
  (ground-truth vs. recovered weights) for each synthetic participant.
- `analyze.py` — summarizes results across all recovery attempts in a
  results directory: a per-parameter true-vs-recovered scatter grid
  (`recovery.png`) and a summary CSV (Pearson r, bias, RMSE per parameter).
- `submit_recovery.sh` — SLURM array submission script for `fast_recover.py`
  (one task per synthetic participant; works for a single task or many
  running concurrently).

## Other files

- `utils.py` — board/feature math: turns a small pattern template (like
  "4 in a row") into the full list of concrete board positions that match
  it, builds a heuristic from control params + templates, and converts
  between an older BADS parameterization and this one.
- `fourbynine.py` / `_swig_fourbynine.so` — the auto-generated Python
  interface to the C++ engine. Don't hand-edit these.

## Running the tests

Tests for this code live at `../tests/python/`, not inside `model_fitting/`
itself:

- `check_installation.py`, `conftest.py` — check your environment/install
  is set up correctly.
- `feature_test.py`, `search_test.py` — check the model's features and
  search behave correctly.
- `test_checkpoint_resume.py` — checks that pausing and resuming a fit
  produces the same result as not pausing at all.
- `timing_test.py`, `profile_singlethreaded_fitter.py`,
  `ibs_variance_experiment.py`, `demonstrate_noise_equivalence.py`,
  `test_fit_monitor_dashboard.py` — performance and behavior diagnostics,
  not really "does this pass or fail" tests.

Run them with:

```
cd model_fitting
python -m pytest ../tests/python/
```

## A note on `monkey_4iar`

A separate project, `monkey_4iar`, imports `tree_search.py` and
`tree_search_fitter.py` directly from this repo and builds its own
production settings on top of them. There's no formal contract between the
two projects — if you change how `TreeSearch` is built or how
`MultiThreadedFitter` is called, it's worth checking whether `monkey_4iar`
depends on the old behavior before you do.
