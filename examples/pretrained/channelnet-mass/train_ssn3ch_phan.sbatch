#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --job-name=ssn3ch-phan
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# SeqSleepNet 3ch on the clean channels with the phan protocol (Adam 1e-4 constant, no dropout/wd, stride 1,
# 10 passes, selection by validation loss): the protocol-matched baseline of the multimodal S3-F (#25).
# Env: CODE (checkout of feat/channelnet-reg), OUT (run folder).
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python -m physioex.train.bin.train_channelnet_mm --arch ssn3ch --stage S0 --root /sofia/projects/2026_101/cache/v1 \
  --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05 \
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --num-workers 8 \
  --passes 10 --select loss --sched const --issue xmai-lab/sleep-mechinterp#25 --out $OUT
