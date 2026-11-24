#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=channelnet-mm-reg
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# reg123 protocol (xmai-lab/sleep-mechinterp#21) on the two multichannel models, MASS SS01-SS05 fold 0.
#   task 0: SeqSleepNet 3ch baseline on the clean channels, from scratch (S0)
#   task 1: multimodal ChannelNet from the reg123 branches: S2-F (branches frozen), then S3-F from it
# Env: CODE (checkout of feat/channelnet-reg), RUN (run folder), BRANCHES (folder with reg123_{EEG,EOG,EMG}/).
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
TRAIN="python -m physioex.train.bin.train_channelnet_mm"
COMMON=(--root /sofia/projects/2026_101/cache/v1
  --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --num-workers 8
  --dropout 0.1 --wd 1e-4 --sampler random --max-shift 300 --steps 48760 --val-every 487
  --select kappa --patience 20 --issue xmai-lab/sleep-mechinterp#24)
case $SLURM_ARRAY_TASK_ID in
  0) $TRAIN --arch ssn3ch --stage S0 --sched cosine "${COMMON[@]}" --out $RUN/ssn3ch ;;
  1) MM=(--branches $BRANCHES --branch-prefix reg123_ --points F)
     $TRAIN --stage S2 --sched cosine "${MM[@]}" "${COMMON[@]}" --out $RUN/S2_F || exit 1
     $TRAIN --stage S3 --init $RUN/S2_F "${MM[@]}" "${COMMON[@]}" --out $RUN/S3_F ;;
esac
