#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=fix-alz
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
export MODELS_DIR=/leonardo_work/IscrC_SELFEEG/pretrained

OUTDIR=/leonardo_work/IscrC_SELFEEG/pretrained/fix-alzheimer
mkdir -p "$OUTDIR"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

python examples/pretrained/protosleepnet-gagliardi/fix-alzheimer/train_roto_translation.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --models_dir "$MODELS_DIR" \
    --lr 1e-3 \
    --max_epochs 50 \
    --patience 10 \
    --batch_size 32
