#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=seed-mmd
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%A_%a.out
#SBATCH --array=0-1

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

cd /leonardo_work/IscrC_SELFEEG/code/physioex

SCRIPT=examples/pretrained/protosleepnet-gagliardi/seed_mmd_analysis.py

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== PSN (SEQ/MASS) ==="
    python $SCRIPT \
        --results_dir /leonardo_work/IscrC_SELFEEG/results/seed_stability/seq \
        --backbone_name PSN \
        --seeds 123 456 789 1024 2048 3141
else
    echo "=== PST (ST/SHHS) ==="
    python $SCRIPT \
        --results_dir /leonardo_work/IscrC_SELFEEG/results/seed_stability/st \
        --backbone_name PST \
        --seeds 123 456 789 1024 2048 3141
fi
