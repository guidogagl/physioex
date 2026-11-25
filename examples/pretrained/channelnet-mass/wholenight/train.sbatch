#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --array=0-41%12
#SBATCH --job-name=wholenight-mass
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# Whole-night sequence fine-tuning of MASS S3-F (xmai-lab/sleep-mechinterp#34): 14 configurations x 3 seeds,
# task i -> configuration CONFIG_LIST[i // 3], seed i % 3. Env: CODE, RUN, INIT (S3-F run folder); extra args via ARGS
# (e.g. --no-train for the parity check).
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
CFG=$(python -c "from physioex.train.bin.train_wholenight import CONFIG_LIST as C; print(C[$SLURM_ARRAY_TASK_ID // 3])")
SEED=$((SLURM_ARRAY_TASK_ID % 3))
python -m physioex.train.bin.train_wholenight --config $CFG --seed $SEED --init $INIT \
  --root /sofia/projects/2026_101/cache/v1 --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05 \
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --out ${OUT:-$RUN/${CFG}_s${SEED}} $ARGS
