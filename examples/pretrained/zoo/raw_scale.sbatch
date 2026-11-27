#!/bin/bash
#SBATCH --partition=zen5_vis
#SBATCH --account=vsc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=zoo-rawscale
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# Raw-input scale calibration (xmai-lab/sleep-mechinterp#40). Env: CODE, OUT.
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=$CODE
cd $CODE
python examples/pretrained/zoo/raw_scale_stats.py --root /sofia/projects/2026_101/cache/v1 --out $OUT/raw_scale.json
