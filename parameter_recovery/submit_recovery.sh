#!/bin/bash
# Round-trip recovery smoke test (fast_recover.py), one task per synthetic
# participant. Works for a single participant (--array=0-0) or many run
# concurrently (--array=0-29) -- same script either way.
#
# Usage:
#   sbatch --array=0-29 submit_recovery.sh [--n-boards 30 ...]
#
# Array size must match --n-participants (default 30 in fast_recover.py).
#
# --qos=short requires --time >= 01:15:00 or sbatch silently falls back to a
# more restrictive QOS (MaxJobsPU=2) instead of erroring.
#
#SBATCH -A griffith
#SBATCH --qos=short
#SBATCH -J ts_fast_recovery
#SBATCH --cpus-per-task=4
#SBATCH --time=01:15:00
#SBATCH --mem-per-cpu=900M
#SBATCH -o logs/fast_recovery_%A_%a.log
#SBATCH -e logs/fast_recovery_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# n-workers follows the SLURM allocation.
exec python fast_recover.py --index "${SLURM_ARRAY_TASK_ID}" \
    --n-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
