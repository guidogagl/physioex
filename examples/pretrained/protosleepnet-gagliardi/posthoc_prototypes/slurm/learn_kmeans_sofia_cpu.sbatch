#!/bin/bash
#SBATCH --partition=zen5_dense
#SBATCH --cpus-per-task=8
#SBATCH --account=pilot_2026_0042
#SBATCH --time=00:30:00
#SBATCH --job-name=vq-km
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex-pip/bin/activate

export PYTHONUNBUFFERED=1

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

MODEL_NAME="${MODEL_NAME:-seqsleepnet-phan}"
M="${M:-48}"
EMB_DIR="/sofia/scratch/pilot/pilot_2026_0042/WORK/posthoc_embeddings/${MODEL_NAME}"

echo "K-Means codebook: model=$MODEL_NAME, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
    --emb_dir $EMB_DIR \
    --n_prototypes $M \
    --output_dir $EMB_DIR

echo "Done."
