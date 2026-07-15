#!/bin/bash
# Minimal round-trip recovery smoke test (fast_recover.py) as a single SLURM job.
# Small board set (data/sample) + reduced BADS budget -- sized to finish in a
# couple of minutes on a compute node, not a login node.
#
# Usage:
#   sbatch submit_fast_recovery.sh [--n-boards 40 ...]
#
#SBATCH -A griffith
#SBATCH -J ts_fast_recovery
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=900M
#SBATCH -o logs/fast_recovery_%j.log
#SBATCH -e logs/fast_recovery_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# n-workers follows the SLURM allocation.
exec python fast_recover.py --n-workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
