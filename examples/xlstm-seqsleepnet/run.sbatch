#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
# zen4_h200 (Sep 2026): the old `--gres=gpu:nvidia_h200:1` form is rejected; the partition wants
# --nodes/--ntasks-per-node/--gpus-per-node (typed) and exactly 24 CPUs per GPU. --mem is not allowed.
#SBATCH --job-name=xlstm-seqsleepnet
#SBATCH --output=/sofia/projects/2026_101/WORK/logs/xlstm-seqsleepnet/%x_%A_%a.out
#
# One run.py cell per job. Arguments come from the environment so the same
# script serves every stage:
#   RUN_ARGS   the full run.py argument string (required)
#   REPO       physioex checkout (default $WORK/physioex)
#   OUT_DIR    default $OUTPUTS/xlstm-seqsleepnet
# Array usage (one fold per task):
#   sbatch --array=0-19 --export=ALL,RUN_ARGS="--dataset mass --model seqsleepnet ... --fold \$SLURM_ARRAY_TASK_ID" run.sbatch
# (escape the \$ so it expands inside the job, or pass FOLD_FROM_ARRAY=1)

set -euo pipefail

# login nodes may have no cluster module preloaded (swap fails) -> load, then swap as fallback
module load cluster/sofia/zen4_h200 2>/dev/null || module swap cluster/sofia/zen4_h200 2>/dev/null || true
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load matplotlib/3.10.3-gfbf-2025a scikit-learn/1.7.0-gfbf-2025a
module load h5py/3.14.0-foss-2025a PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0 Pillow/11.3.0-GCCcore-14.2.0
source /sofia/projects/2026_101/WORK/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Force the 2026_101 paths: with --export=ALL the login profile's PHYSIOEX_CACHE_DIR (still the
# full, read-only pilot_2026_0042 fileset) would win over a `${VAR:-default}`; raw data are passed
# to run.py via --dataset_root (they only exist on the pilot scratch, read-only).
export PHYSIOEX_CACHE_DIR="${CACHE_DIR:-/sofia/projects/2026_101/cache}"
export PHYSIOEX_DATA="${DATA_DIR:-/sofia/projects/2026_101/raw_data}"

REPO="${REPO:-/sofia/projects/2026_101/WORK/physioex}"
OUT_DIR="${OUT_DIR:-/sofia/projects/2026_101/WORK/outputs/xlstm-seqsleepnet}"
# the venv's editable physioex may point at another checkout: import from $REPO first.
# PYLIB: vendored pure-python packages (xlstm, mlstm_kernels, safetensors, einops) unpacked
# from wheels — the job venv lives on a full, read-only fileset and cannot be pip-installed.
PYLIB="${PYLIB:-/sofia/projects/2026_101/WORK/pylib}"
export PYTHONPATH="${REPO}:${PYLIB}${PYTHONPATH:+:$PYTHONPATH}"
# RUN_ARGS_FILE: read the arguments from a file (sbatch --export splits on commas, which
# breaks JSON kwargs). {fold} inside the arguments is replaced by the array task id.
if [[ -n "${RUN_ARGS_FILE:-}" ]]; then RUN_ARGS="$(cat "$RUN_ARGS_FILE")"; fi
: "${RUN_ARGS:?set RUN_ARGS (or RUN_ARGS_FILE) to the run.py arguments}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_ARGS="${RUN_ARGS//\{fold\}/${SLURM_ARRAY_TASK_ID}}"
  if [[ "${FOLD_FROM_ARRAY:-0}" == "1" ]]; then RUN_ARGS="${RUN_ARGS} --fold ${SLURM_ARRAY_TASK_ID}"; fi
fi
# {seed} is replaced by $SEED (default 0), so one argument file serves every seed
RUN_ARGS="${RUN_ARGS//\{seed\}/${SEED:-0}}"

mkdir -p "$OUT_DIR" "$(dirname "/sofia/projects/2026_101/WORK/logs/xlstm-seqsleepnet/x")"
cd "$REPO"
echo "[sbatch] $(date) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[sbatch] commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
# fail fast if the environment is wrong (torch from the module, physioex from $REPO)
python -c "import torch, physioex, sys; print(f'[sbatch] python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()} physioex from {physioex.__file__}')"
echo "[sbatch] run.py ${RUN_ARGS}"

# shellcheck disable=SC2086
python examples/xlstm-seqsleepnet/run.py ${RUN_ARGS} --out_dir "$OUT_DIR" --gpu_id 0 --num_workers "${NUM_WORKERS:-8}"
