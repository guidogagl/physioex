#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=seed-allm
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%A_%a.out
#SBATCH --array=0-5

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

# Copy MASS cache to /dev/shm
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
for c in 01 02 03 04 05; do
    cp -r $CACHE_SRC/v1/mass_ss${c} /dev/shm/physioex_cache/v1/ 2>/dev/null
done
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache

cd /leonardo_work/IscrC_SELFEEG/code/physioex

RESULTS=/leonardo_work/IscrC_SELFEEG/results/seed_stability/seq
SEEDS=(123 456 789 1024 2048 3141)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID → seed $S, all M values"

python examples/pretrained/protosleepnet-gagliardi/seed_recon_all_m.py \
    --seed_dir $RESULTS/seed_$S \
    --backbone seq --gpu_id 0 --top_k 256
