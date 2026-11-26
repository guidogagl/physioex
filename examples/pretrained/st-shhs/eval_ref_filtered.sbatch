#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --job-name=st-ref-filt
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# Reference (#28) re-scored on the five-stage-filtered test split (1532 nights) of the framework runs.
# Env: CODE, OUT.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $CODE
for S in $(seq 0 250 1750); do                    # fresh process per chunk: the dataset leaks open files across calls
  python examples/pretrained/st-shhs/eval_ref_filtered.py --out $OUT --start $S --stop $((S + 250)) || exit 1
done
python examples/pretrained/st-shhs/eval_ref_filtered.py --out $OUT --merge
