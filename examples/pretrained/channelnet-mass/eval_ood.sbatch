#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --array=0-19%6
#SBATCH --job-name=ood-mass
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# Out-of-domain evaluation (#25), one array task per target dataset; results in $RUN/<dataset>.json.
# Env: CODE (checkout of feat/channelnet-reg), RUN (output folder), MODELS (name=run_folder ..., reference first).
DS=(alzheimers_ad alzheimers_hc ccshs dcsm dreams hmc hpap lsec mesa mros parkinsons_nap_hoa parkinsons_nap_pd
    parkinsons_night_hoa parkinsons_night_pd shhs_visit1 shhs_visit2 wsc_visit1 wsc_visit2 wsc_visit3 wsc_visit4)
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
python examples/pretrained/channelnet-mass/eval_ood.py --models $MODELS --root /sofia/projects/2026_101/cache/v1 \
  --datasets ${DS[$SLURM_ARRAY_TASK_ID]} --max-nights 300 --out $RUN
