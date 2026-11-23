#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --job-name=ssn-phan-3ch
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# 3-channel SeqSleepNet baseline (raw EEG, EOG, EMG) on MASS SS01-05, seqsleepnet-phan protocol. Needs CODE and RUN.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$CODE
cd $CODE
python examples/pretrained/channelnet-mass/train_seqsleepnet_3ch.py --gpu_id 0 --output_dir $RUN/seqsleepnet-phan-3ch \
  --dataset_root /sofia/projects/2026_101/raw_data/MASS/Original --num_workers 8
