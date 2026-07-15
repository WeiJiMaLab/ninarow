#!/bin/bash
# Parameter-recovery array job: one task per ground-truth point.
# Cluster-agnostic -- set account/partition via env or edit below. Designed to run
# off Della (e.g. NYU torch) to save Della compute.
#
# Usage:
#   sbatch --array=0-14 [--account=...] submit_recovery.sh \
#       --ground-truth ground_truth.jsonl \
#       --boards /scratch/.../modeling_2500 \
#       --out-dir results
#
# Array size must match the number of lines in the ground-truth file
# (15 for the per-fold TreeSearch fits).
#
#SBATCH -J ts_param_recovery
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=900M
#SBATCH -o logs/recovery_%A_%a.log
#SBATCH -e logs/recovery_%A_%a.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# n_workers follows the SLURM allocation (recover.py reads SLURM_CPUS_PER_TASK).
exec python recover.py --jobid "${SLURM_ARRAY_TASK_ID}" "$@"
