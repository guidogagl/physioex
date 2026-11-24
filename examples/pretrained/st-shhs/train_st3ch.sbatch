#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --job-name=st3ch-shhs
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# SleepTransformer 3ch baseline on the clean EEG, EOG, EMG of SHHS-1, paper input (per-frequency z-score) and paper
# protocol, from scratch (S0): the analogue of ssn3ch_phan (#25). Env: CODE, OUT; extra args via ARGS.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
LOCAL=/tmp/$USER-$SLURM_JOB_ID; trap 'rm -rf $LOCAL' EXIT
python -m physioex.train.bin.train_channelnet_mm --arch st3ch --stage S0 --root /sofia/projects/2026_101/cache/v1 \
  --datasets shhs_visit1 --split-file examples/pretrained/st-shhs/shhs1_split.json \
  --local-dir $LOCAL --workers 20 --num-workers 4 \
  --seq-len 21 --batch 32 --eps 1e-7 --dropout 0.1 --steps 1000000 --val-every 100 --patience 200 --min-vals 5000 \
  --select loss --sched const --issue xmai-lab/sleep-mechinterp#29 --out $OUT $ARGS
