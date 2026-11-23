#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --job-name=channelnet-mass
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# ChannelNet (EEG, EOG, EMG: one array task each) on MASS SS01-SS05, protocol of seqsleepnet-phan, fold 0.
# Needs CODE (checkout of feat/channelnet-clean) and RUN (run folder) in the environment.
MODS=(EEG EOG EMG); M=${MODS[$SLURM_ARRAY_TASK_ID]}
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python -m physioex.train.bin.train_channelnet --modality $M --protocol phan \
  --root /sofia/projects/2026_101/cache/v1 --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05 \
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --num-workers 8 \
  --issue xmai-lab/sleep-mechinterp#16 --out $RUN/$M
