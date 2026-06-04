#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=pilot_2026_0042
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke-prosleepnet
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

module swap cluster/sofia/zen4_h200 2>/dev/null
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load matplotlib/3.10.3-gfbf-2025a scikit-learn/1.7.0-gfbf-2025a
module load h5py/3.14.0-foss-2025a PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0 Pillow/11.3.0-GCCcore-14.2.0
source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

OUTDIR="/sofia/scratch/pilot/pilot_2026_0042/WORK/pretrained/smoke/protosleepnet-gagliardi"
mkdir -p "$OUTDIR"

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

python examples/pretrained/protosleepnet-gagliardi/train.py \
    --gpu_id 0 \
    --max_epochs 1 \
    --output_dir "$OUTDIR" \
    --datasets sleepedf
