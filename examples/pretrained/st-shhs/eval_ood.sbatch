#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --array=0-22%8
#SBATCH --job-name=ood-shhs
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# SleepTransformer framework trained on SHHS-1 (xmai-lab/sleep-mechinterp#28-#30), scored with eval_ood.py.
#   out of domain (#30): one array task per target cohort (SHHS excluded, MASS added), 300 nights at most
#   in domain (#28/#29): INDOMAIN=1, --array=0, all 1532 test nights of shhs1_split.json
# Env: CODE, RUN (output folder), MODELS (name=run_folder ..., reference first), INDOMAIN (optional).
DS=(alzheimers_ad alzheimers_hc ccshs dcsm dreams hmc hpap lsec mesa mros parkinsons_nap_hoa parkinsons_nap_pd
    parkinsons_night_hoa parkinsons_night_pd wsc_visit1 wsc_visit2 wsc_visit3 wsc_visit4
    mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05)
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
if [ -n "$INDOMAIN" ]; then
  python examples/pretrained/channelnet-mass/eval_ood.py --models $MODELS --root /sofia/projects/2026_101/cache/v1 \
    --datasets shhs_visit1 --split-file examples/pretrained/st-shhs/shhs1_split.json --max-nights 0 --out $RUN \
    --issue xmai-lab/sleep-mechinterp#29
else
  python examples/pretrained/channelnet-mass/eval_ood.py --models $MODELS --root /sofia/projects/2026_101/cache/v1 \
    --datasets ${DS[$SLURM_ARRAY_TASK_ID]} --max-nights 300 --out $RUN --issue xmai-lab/sleep-mechinterp#30
fi
