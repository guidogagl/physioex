#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2
#SBATCH --job-name=st-branches-shhs
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# SleepTransformer branches (EEG, EOG, EMG: one array task each) on SHHS-1 with the paper protocol (st-phan:
# L 21, Adam 1e-4 eps 1e-7, batch 32, dropout 0.1, validation every 100 steps, stop after 200 validations without
# improvement once 5,000 validations are done). Nights are copied to the node's /tmp and memory-mapped from there.
# Env: CODE (checkout of feat/st-framework), RUN (run folder). Extra args (e.g. --steps for the I/O check) via ARGS.
MODS=(EEG EOG EMG); M=${MODS[$SLURM_ARRAY_TASK_ID]}
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
LOCAL=/tmp/$USER-$SLURM_JOB_ID; trap 'rm -rf $LOCAL' EXIT
python -m physioex.train.bin.train_channelnet --arch sleeptransformer --protocol st-phan --modality $M \
  --root /sofia/projects/2026_101/cache/v1 --datasets shhs_visit1 \
  --split-file examples/pretrained/st-shhs/shhs1_split.json --local-dir $LOCAL --workers 20 --num-workers 4 \
  --issue xmai-lab/sleep-mechinterp#28 --out $RUN/$M $ARGS
