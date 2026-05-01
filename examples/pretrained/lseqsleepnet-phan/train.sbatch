#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nice=200
#SBATCH --time=48:00:00
#SBATCH --job-name=lseqslp-phan
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0012/logs/pretrained/%x_%j.out

PROJECT=/sofia/scratch/pilot/pilot_2026_0012
GUIDO="${PROJECT}/projects/Guido"

module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0 \
            SciPy-bundle/2025.06-gfbf-2025a \
            matplotlib/3.10.3-gfbf-2025a \
            scikit-learn/1.7.0-gfbf-2025a \
            PyYAML/6.0.2-GCCcore-14.2.0 \
            ml_dtypes/0.5.1-gfbf-2025a

source "${GUIDO}/physioex-env/bin/activate"

export PHYSIOEX_CACHE_DIR="${PROJECT}/data/Guido/.cache/physioex"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

OUTDIR="${PROJECT}/data/Guido/pretrained/lseqsleepnet-phan"
mkdir -p "$OUTDIR"
mkdir -p "${PROJECT}/logs/pretrained"

cd "${GUIDO}/physioex"

python examples/pretrained/lseqsleepnet-phan/train.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --dataset_root "${PROJECT}/data/Guido/shhs"
