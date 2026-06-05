#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

# Usage: sbatch --export=M=48 --job-name=train-seq-proto-m48 train_seq_proto_leonardo.sbatch

# Use torch2x venv (PyTorch 2.4+cu121) — required for SimVQ/einx
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy SHHS cache to /dev/shm for fast I/O
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
echo "Copying shhs_visit1 cache to /dev/shm..."
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

echo "Training SEQ protosleepnet with M=${M} prototypes"

python examples/pretrained/protosleepnet-gagliardi/ablation/train_ablation_seqsleepnet.py \
    --variant protosleepnet \
    --n_prototypes ${M} \
    --datasets shhs \
    --gpu_id 0 \
    --num_workers 14 \
    --output_dir /leonardo_work/IscrC_SELFEEG/pretrained/seq-protosleepnet-m${M}
