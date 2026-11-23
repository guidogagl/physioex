#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --job-name=ssn-baselines-mass
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# SeqSleepNet baselines (seqsleepnet-phan 1ch, physioex 1.x 3ch) on the MASS fold-0 test set. Needs CODE and RUN.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python examples/pretrained/channelnet-mass/eval_baselines.py --weights /sofia/projects/2026_101/models/weights/seqsleepnet-baselines \
  --root /sofia/projects/2026_101/raw_data/MASS/Original --out $RUN/baselines
