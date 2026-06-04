#!/bin/bash
#SBATCH --job-name=seqsleep-shhs-3ch
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nice=200
#SBATCH --account=pilot_2026_0042
#SBATCH --time=02:00:00
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

module swap cluster/sofia/zen4_h200 2>/dev/null
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load matplotlib/3.10.3-gfbf-2025a scikit-learn/1.7.0-gfbf-2025a
module load h5py/3.14.0-foss-2025a PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0 Pillow/11.3.0-GCCcore-14.2.0
source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

python examples/pretrained/seqsleepnet-phan/train_shhs.py \
    --gpu_id 0 \
    --num_workers 20 \
    --output_dir /sofia/scratch/pilot/pilot_2026_0042/WORK/checkpoints/seqsleepnet-shhs-3ch
