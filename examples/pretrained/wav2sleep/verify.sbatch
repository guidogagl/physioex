#!/bin/bash
# Reproduce the published wav2sleep results on a Slurm GPU node.
# Cluster-specific options (account, partition, GPU type) are given at submit time, e.g.
#   sbatch -A <account> -p <gpu-partition> --gpus-per-node=1 examples/pretrained/wav2sleep/verify.sbatch [verify.py args]
# Needs PHYSIOEX_DATA (raw NSRR data root) and, optionally, PHYSIOEX_CACHE_DIR and VENV (a virtualenv with physioex).
#SBATCH --job-name=verify-wav2sleep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=verify-wav2sleep-%j.out
set -euo pipefail
: "${PHYSIOEX_DATA:?set PHYSIOEX_DATA to the raw data root}"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${SLURM_SUBMIT_DIR:-.}"
python examples/pretrained/wav2sleep/verify.py --out "verify-wav2sleep-${SLURM_JOB_ID:-local}.json" "$@"
