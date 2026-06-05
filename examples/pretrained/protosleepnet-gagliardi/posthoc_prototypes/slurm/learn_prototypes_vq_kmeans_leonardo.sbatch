#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=vq-kmeans
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1

cd /leonardo_work/IscrC_SELFEEG/code/physioex

MODEL="${MODEL:-st-baseline}"
M="${M:-48}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL

echo "VQ K-Means: model=$MODEL, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
    --emb_dir $EMB_DIR \
    --n_prototypes $M \
    --output_dir $EMB_DIR
