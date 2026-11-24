#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --job-name=channelnet-reg
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# Regularisation ablation (xmai-lab/sleep-mechinterp#23): {reg1, reg12, reg123} x {EEG, EOG, EMG}, MASS fold 0.
# Needs CODE (checkout of feat/channelnet-reg) and RUN (run folder) in the environment.
MODS=(EEG EOG EMG); PROTS=(reg1 reg12 reg123)
M=${MODS[$((SLURM_ARRAY_TASK_ID % 3))]}; PR=${PROTS[$((SLURM_ARRAY_TASK_ID / 3))]}
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python -m physioex.train.bin.train_channelnet --modality $M --protocol $PR \
  --root /sofia/projects/2026_101/cache/v1 --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05 \
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --num-workers 8 \
  --issue xmai-lab/sleep-mechinterp#23 --out $RUN/${PR}_$M
