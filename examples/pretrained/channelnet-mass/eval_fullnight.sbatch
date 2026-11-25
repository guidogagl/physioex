#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --job-name=fullnight-mass
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# Full-night vs voting inference (#31). Env: CODE (checkout), OUT (output folder), MODEL (multimodal run folder).
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python examples/pretrained/channelnet-mass/eval_fullnight.py --model $MODEL --root /sofia/projects/2026_101/cache/v1 \
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --out $OUT
