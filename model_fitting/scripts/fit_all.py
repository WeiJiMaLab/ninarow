"""Interactive multistart CV fitting driver, over a grid of participants x CV folds.

data_dir is a parent directory containing one subdirectory per participant, each
holding n_splits fold CSVs (0.csv..N-1.csv), e.g.:

    data_dir/
      participant1/0.csv 1.csv 2.csv
      participant2/0.csv 1.csv 2.csv

Total jobs = n_starts x n_splits x n_participants (one multistart-fit-and-reduce per
(participant, held-out fold)). Detects participants + validates each has exactly
n_splits fold files, then asks whether to fit sequentially (on this machine) or in
parallel (submits a SLURM array job + a dependent reduce job per grid point). Either
way, each grid point ends with a winner selected by re-evaluating every start's params
at high IBS repeats and picking the argmin (mirrors monkey_4iar's MultiRunner.select —
avoids picking a start that just got lucky noise).
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multistart import default_model_factory, fit_one_start, select_winner, write_result_json, write_start_json
from fit_one_start import check_data_dir, load_split

MODEL_FITTING_DIR = Path(__file__).resolve().parent.parent

_USE_COLOR = sys.stdout.isatty()
_CODES = {
    "bold": "1", "dim": "2",
    "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36",
}


def c(text, *styles):
    """Wrap text in ANSI codes for the given style names (see _CODES). No-op when
    stdout isn't a TTY (piped/redirected output stays plain, e.g. into a log file)."""
    if not _USE_COLOR:
        return text
    prefix = "".join(f"\033[{_CODES[s]}m" for s in styles)
    return f"{prefix}{text}\033[0m"


def success(text):
    return f"{c('[Success]', 'green', 'bold')} {text}"


def failure(text):
    return f"{c('[Failure]', 'red', 'bold')} {text}"


def prompt_choice(question, options):
    print(f"\n{c('[Query]', 'cyan', 'bold')}\n{c(question, 'bold')}")
    for i, opt in enumerate(options, 1):
        print(f"  {c(str(i) + ')', 'yellow')} {opt}")
    while True:
        raw = input(f"{c('>', 'cyan', 'bold')} ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(c(f"Please enter a number 1-{len(options)}.", "red"))


def prompt_yes_no(question):
    while True:
        raw = input(f"{c(question, 'bold')} {c('(y/n)', 'dim')} ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(c("Please enter y or n.", "red"))


def prompt_text(question, default):
    raw = input(f"{c(question, 'bold')} {c(f'[{default}]', 'dim')}: ").strip()
    return raw or default


def prompt_int(question, default):
    while True:
        raw = input(f"{c(question, 'bold')} {c(f'[{default}]', 'dim')}: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            return int(raw)
        print(c("Please enter a whole number.", "red"))


def detect_participants(data_dir, n_splits):
    """Subdirectories of data_dir that each contain exactly n_splits fold CSVs."""
    participants = []
    for child in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        try:
            check_data_dir(child, n_splits)
        except (FileNotFoundError, ValueError):
            continue
        participants.append(child)
    return participants


def run_one_grid_point_sequential(participant_dir, n_splits, held_out_index, n_starts,
                                   n_workers, n_repeats, verbose):
    train, test = load_split(check_data_dir(participant_dir, n_splits), held_out_index)
    starts_dir = participant_dir / "starts"
    starts_dir.mkdir(exist_ok=True)

    starts = []
    for start in range(n_starts):
        model = default_model_factory(verbose=verbose)()
        record = fit_one_start(
            model, train, start, n_workers=n_workers, n_repeats=n_repeats, verbose=verbose,
        )
        write_start_json(starts_dir / f"{held_out_index}.{start}.json", record)
        print(f"  [{participant_dir.name}] fold {held_out_index} start {start}: "
              f"train_nll_raw={record['train_nll_raw']:.4f}")
        starts.append(record)

    winner_idx, winner, train_nll, test_nll = select_winner(
        default_model_factory(verbose=verbose), starts, train, test,
        n_workers=n_workers, verbose=verbose,
    )
    result_path = participant_dir / "results" / f"{held_out_index}.json"
    write_result_json(result_path, held_out_index, winner, train_nll, test_nll, n_starts, model.param_names)
    print(success(
        f"[{participant_dir.name}] fold {held_out_index} winner: start {winner['start']} "
        f"({winner_idx + 1}/{n_starts}) train_nll={sum(train_nll):.4f} test_nll={sum(test_nll):.4f}"
    ))
    print(f"  [{participant_dir.name}] fold {held_out_index} results written to {result_path}")
    return winner, train_nll, test_nll


def run_sequential(participants, n_splits, n_starts, n_workers, n_repeats, verbose):
    results = {}
    total = len(participants) * n_splits
    done = 0
    for participant_dir in participants:
        for held_out_index in range(n_splits):
            done += 1
            print(c(f"\n--- Grid point {done}/{total}: "
                     f"{participant_dir.name} fold {held_out_index} ---", "magenta", "bold"))
            winner, train_nll, test_nll = run_one_grid_point_sequential(
                participant_dir, n_splits, held_out_index, n_starts, n_workers, n_repeats, verbose,
            )
            results[(participant_dir.name, held_out_index)] = (winner, train_nll, test_nll)

    print(c("\n=== Summary ===", "yellow", "bold"))
    for (name, fold), (winner, train_nll, test_nll) in results.items():
        print(f"{name} fold {fold}: train_nll={sum(train_nll):.4f} test_nll={sum(test_nll):.4f}")
    print("\nResults written to <participant_dir>/results/<held_out_index>.json for each grid point, e.g.:")
    for participant_dir in participants:
        print(f"  {participant_dir / 'results'}/")
    return results


def suggest_fit_time(n_repeats):
    """Rough sbatch --time suggestion for one fit-array task (one BADS start). Scales
    with n_repeats (dominant cost driver: IBS repeats per BADS function eval) off a
    ~1-day baseline at n_repeats=40 (the monkey_4iar production ramp top); floored at
    1h and capped at 7 days (Della's usual max wall time). This is a suggestion only —
    always user-overridable, since actual runtime also depends on dataset size and
    per-fold trial counts this script has no visibility into at prompt time."""
    hours = max(1.0, 24.0 * (n_repeats / 40.0))
    hours = min(hours, 24 * 7)
    return format_hms(hours)


def format_hms(hours):
    total_minutes = round(hours * 60)
    h, m = divmod(total_minutes, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}-{h:02d}:{m:02d}:00"
    return f"{h:02d}:{m:02d}:00"


def run_parallel(participants, n_splits, n_starts, n_repeats, account, cores_per_job, fit_time):
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    submitted = []
    for participant_dir in participants:
        participant_dir = participant_dir.resolve()
        (participant_dir / "starts").mkdir(exist_ok=True)
        for held_out_index in range(n_splits):
            fit_cmd = [
                "sbatch",
                f"--account={account}",
                f"--cpus-per-task={cores_per_job}",
                f"--time={fit_time}",
                f"--array=0-{n_starts - 1}",
                "--job-name", f"ninarow_fit_{participant_dir.name}_{held_out_index}",
                f"--output={logs_dir}/multistart_%A_%a.log",
                f"--error={logs_dir}/multistart_%A_%a.log",
                "--parsable",
                "--wrap",
                (
                    f"cd {MODEL_FITTING_DIR} && "
                    f"python scripts/fit_one_start.py {participant_dir} {n_splits} "
                    f"{held_out_index} --n-repeats {n_repeats} --n-workers {cores_per_job}"
                ),
            ]
            try:
                fit_job_id = subprocess.check_output(
                    fit_cmd, text=True, stderr=subprocess.STDOUT
                ).strip().split(";")[0]
            except subprocess.CalledProcessError as e:
                print(failure(
                    f"[{participant_dir.name}] fold {held_out_index}: "
                    f"fit array submission FAILED, skipping this grid point:\n{e.output}"
                ))
                submitted.append((participant_dir.name, held_out_index, None, None))
                continue

            reduce_cmd = [
                "sbatch",
                f"--account={account}",
                f"--dependency=afterok:{fit_job_id}",
                "--time=00:30:00",
                "--cpus-per-task=4",
                "--mem-per-cpu=2G",
                "--job-name", f"ninarow_reduce_{participant_dir.name}_{held_out_index}",
                f"--output={logs_dir}/reduce_%j.log",
                f"--error={logs_dir}/reduce_%j.log",
                "--parsable",
                "--wrap",
                (
                    f"cd {Path(__file__).resolve().parent.parent} && "
                    f"python scripts/consolidate.py {participant_dir} {n_splits} "
                    f"{held_out_index} {n_starts} --n-workers 4"
                ),
            ]
            try:
                reduce_job_id = subprocess.check_output(
                    reduce_cmd, text=True, stderr=subprocess.STDOUT
                ).strip()
            except subprocess.CalledProcessError as e:
                print(failure(
                    f"[{participant_dir.name}] fold {held_out_index}: "
                    f"fit array {fit_job_id} submitted OK, but reduce submission FAILED:\n{e.output}"
                ))
                submitted.append((participant_dir.name, held_out_index, fit_job_id, None))
                continue

            print(success(
                f"[{participant_dir.name}] fold {held_out_index}: "
                f"fit array {fit_job_id} ({n_starts} starts) -> reduce {reduce_job_id}"
            ))
            submitted.append((participant_dir.name, held_out_index, fit_job_id, reduce_job_id))

    ok = [s for s in submitted if s[3] is not None]
    failed = [s for s in submitted if s[3] is None]
    print(c(f"\nSubmitted {len(ok)}/{len(submitted)} (fit array + reduce) job pairs successfully.",
            "green" if not failed else "yellow", "bold"))
    if failed:
        print(failure(f"Failed to fully submit {len(failed)} grid point(s): "
              + ", ".join(f"{name} fold {fold}" for name, fold, _, _ in failed)))
    print(c("Monitor with: squeue -u $USER", "dim"))
    print(c(f"Reduce logs land in: {logs_dir}/reduce_<jobid>.log", "dim"))
    print("\nOnce each reduce job finishes, results land in "
          "<participant_dir>/results/<held_out_index>.json, e.g.:")
    for participant_dir in participants:
        print(f"  {c(str(participant_dir / 'results'), 'cyan')}/")
    return submitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=str, nargs="?", default=None,
                         help="Parent directory of per-participant fold subdirectories. "
                              "Prompted for (default: ../data/sample) if omitted.")
    parser.add_argument("n_splits", type=int, nargs="?", default=None,
                         help="Prompted for (default: 3) if omitted.")
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Sequential path only (default: SLURM_CPUS_PER_TASK if set, else 6).")
    parser.add_argument("--n-repeats", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final confirmation prompt.")
    args = parser.parse_args()

    print(c("Beginning 4IAR fitting ...", "bold"))

    data_dir_raw = args.data_dir
    if data_dir_raw is None:
        data_dir_raw = prompt_text("Data directory (parent of per-participant fold subfolders)",
                                    str(MODEL_FITTING_DIR.parent / "data" / "sample"))
    n_splits = args.n_splits
    if n_splits is None:
        n_splits = prompt_int("Number of CV splits (fold CSVs per participant)", 3)
    args.n_splits = n_splits
    data_dir = Path(data_dir_raw)

    print("Detecting folder ...")
    if not data_dir.is_dir():
        sys.exit(failure(f"Folder not found: {data_dir}"))
    print(success("Folder found!"))

    participants = detect_participants(data_dir, args.n_splits)
    if not participants:
        sys.exit(failure(f"No participant subdirectories with {args.n_splits} CV splits found under {data_dir}"))
    print(success(f"Detected {args.n_splits} CV splits"))
    print(success(f"Detected {len(participants)} participant(s): "
                   + ", ".join(p.name for p in participants)))

    total_jobs = args.n_starts * args.n_splits * len(participants)
    print(c(f"\nThis consists of [{args.n_starts} x {args.n_splits} x {len(participants)}] "
            f"= {total_jobs} jobs", "yellow", "bold"))

    mode = prompt_choice(
        "How should the multi-start fit run?",
        ["Sequentially", "In Parallel"],
    )

    account = None
    cores_per_job = None
    fit_time = None
    if mode == 1:
        account = input(f"\n{c('SLURM account to submit under:', 'bold')} ").strip()
        if not account:
            sys.exit(failure("An account is required for the parallel path."))
        cores_raw = input(f"{c('Cores per job (--cpus-per-task)', 'bold')} {c('[8]', 'dim')}: ").strip()
        cores_per_job = int(cores_raw) if cores_raw else 8
        if cores_per_job <= 0:
            sys.exit(failure(f"Cores per job must be positive, got {cores_per_job}"))
        suggested_time = suggest_fit_time(args.n_repeats)
        time_raw = input(
            f"{c('Time limit per fit job (--time)', 'bold')}, suggested for "
            f"n_repeats={args.n_repeats} {c(f'[{suggested_time}]', 'dim')}: "
        ).strip()
        fit_time = time_raw or suggested_time

    label = lambda s: c(s, "cyan")
    print(f"\n{label('Data directory')} : {data_dir}")
    print(f"{label('Participants  ')} : {len(participants)} ({', '.join(p.name for p in participants)})")
    print(f"{label('CV splits     ')} : {args.n_splits}")
    print(f"{label('Multistarts   ')} : {args.n_starts}")
    print(f"{label('Total jobs    ')} : {c(str(total_jobs), 'yellow', 'bold')}")
    print(f"{label('Mode          ')} : {'Sequential' if mode == 0 else 'Parallel (sbatch)'}")
    if account:
        print(f"{label('SLURM account ')} : {account}")
    if cores_per_job:
        total_cores = total_jobs * cores_per_job
        print(f"{label('Cores per job ')} : {cores_per_job}")
        print(f"{label('Total cores   ')} : {total_jobs} jobs x {cores_per_job} cores "
              f"= {c(str(total_cores), 'yellow', 'bold')} cores")
    if fit_time:
        print(f"{label('Time per job  ')} : {fit_time}")

    if not args.yes and not prompt_yes_no("\nProceed?"):
        print(c("Aborted.", "red"))
        return

    if mode == 0:
        run_sequential(participants, args.n_splits, args.n_starts, args.n_workers, args.n_repeats, args.verbose)
    else:
        run_parallel(participants, args.n_splits, args.n_starts, args.n_repeats, account, cores_per_job, fit_time)


if __name__ == "__main__":
    main()
