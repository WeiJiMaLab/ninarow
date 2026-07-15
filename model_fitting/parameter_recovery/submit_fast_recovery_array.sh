#!/bin/bash
# Round-trip recovery smoke test (fast_recover.py) as a SLURM array job -- one
# task per synthetic participant, run concurrently.
#
# Usage:
#   sbatch --array=0-29 submit_fast_recovery_array.sh [--n-boards 30 ...]
#
# Array size must match --n-participants (default 30 in fast_recover.py).
#
#SBATCH -A griffith
#SBATCH -J ts_fast_recovery
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=900M
#SBATCH -o logs/fast_recovery_%A_%a.log
#SBATCH -e logs/fast_recovery_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# n-workers follows the SLURM allocation.
exec python fast_recover.py --index "${SLURM_ARRAY_TASK_ID}" \
    --n-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
