#!/bin/bash
#SBATCH -J ninarow_multistart
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G
set -euo pipefail

# Generated + submitted by run_multistart.py's parallel path. One array task = one
# multistart start (task id -> start index). Not meant to be edited by hand; rerun
# run_multistart.py to regenerate with different data_dir/n_splits/held_out/n_repeats.
#
# -o/-e are NOT set here as #SBATCH directives: SLURM resolves a relative directive
# path against the submission CWD before this script's own cd ever runs, which crashed
# the job (logs/ didn't exist yet there). run_multistart.py passes absolute
# --output/--error on the sbatch command line instead.
#
# cd via MODEL_FITTING_DIR (exported by run_multistart.py), NOT BASH_SOURCE[0]: under
# sbatch the submitted script is copied to a spool dir (/var/spool/slurmd/...), so
# BASH_SOURCE[0] points there, not at this checkout — dirname'ing it lands nowhere
# near model_fitting/ and python then fails to find scripts/fit_one_start.py at all.
cd "$MODEL_FITTING_DIR"
exec python scripts/fit_one_start.py "$DATA_DIR" "$N_SPLITS" "$HELD_OUT_INDEX" \
    --n-workers "${SLURM_CPUS_PER_TASK:-8}" --n-repeats "$N_REPEATS"
