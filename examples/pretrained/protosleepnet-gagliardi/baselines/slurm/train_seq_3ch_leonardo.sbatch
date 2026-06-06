#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=seq-phan-3ch
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load profile/deeplrn
module load cineca-ai/4.0.0
source /leonardo_work/IscrC_SELFEEG/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy MASS cache to /dev/shm (small: ~1.8G)
echo "Copying MASS cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
for c in 01 02 03 04 05; do
    cp -r $CACHE_SRC/v1/mass_ss${c} /dev/shm/physioex_cache/v1/ 2>/dev/null
done
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

OUTDIR=/leonardo_work/IscrC_SELFEEG/pretrained/seqsleepnet-phan-3ch
mkdir -p "$OUTDIR"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

python examples/pretrained/protosleepnet-gagliardi/baselines/train_seq_3ch.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --num_workers 6
