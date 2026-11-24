#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --job-name=st-mm-shhs
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# Multimodal SleepTransformer on SHHS-1: S2-F (branches frozen, connections + fused head, Adam 1e-3), then S3-F from
# it (new parameters 1e-4, branches 1e-5 as on MASS; warmup 10 % then cosine), both with the paper validation rule.
# Env: CODE, RUN, BRANCHES (folder with EEG/ EOG/ EMG/ best.pt + run.json); extra args via ARGS.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
LOCAL=/tmp/$USER-$SLURM_JOB_ID; trap 'rm -rf $LOCAL' EXIT
TRAIN="python -m physioex.train.bin.train_channelnet_mm"
COMMON=(--branches $BRANCHES --points F --root /sofia/projects/2026_101/cache/v1 --datasets shhs_visit1
  --split-file examples/pretrained/st-shhs/shhs1_split.json --local-dir $LOCAL --workers 20 --num-workers 4
  --seq-len 21 --batch 32 --eps 1e-7 --val-every 100 --patience 200 --min-vals 5000 --select loss
  --issue xmai-lab/sleep-mechinterp#29)
[ -f $RUN/S2_F/best.pt ] || $TRAIN --stage S2 --sched const --steps 1000000 "${COMMON[@]}" --out $RUN/S2_F $ARGS || exit 1
$TRAIN --stage S3 --init $RUN/S2_F --steps 500000 --warmup-frac 0.1 "${COMMON[@]}" --out $RUN/S3_F $ARGS
