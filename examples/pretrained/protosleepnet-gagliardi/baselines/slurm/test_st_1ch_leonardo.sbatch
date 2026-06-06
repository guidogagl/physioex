#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-st-1ch
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load profile/deeplrn
module load cineca-ai/4.0.0
source /leonardo_work/IscrC_SELFEEG/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex

OUTDIR=/leonardo_work/IscrC_SELFEEG/pretrained/sleeptransformer-phan-1ch-eval
mkdir -p "$OUTDIR"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

python examples/pretrained/protosleepnet-gagliardi/baselines/test_st_1ch.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR"
