#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --job-name=st-ref-shhs
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# Reference (#28): the original sleeptransformer-phan (raw C4-A1 EEG, physioex seqsleepnet pipeline) on the SHHS
# benchmark test split, voting L 21. Env: CODE, OUT.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $CODE
python examples/pretrained/sleeptransformer-phan/test_pretrained.py --gpu_id 0 --datasets shhs --output_dir $OUT
