#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=00:25:00
#SBATCH --job-name=vq-test
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy SHHS cache to /dev/shm (test set only ~25GB)
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
export PHYSIOEX_CACHE_DIR=$CACHE_SRC

cd /leonardo_work/IscrC_SELFEEG/code/physioex

MODEL="${MODEL:-st-baseline}"
M="${M:-48}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL
CODEBOOK=$EMB_DIR/codebook_vq_m${M}.npy
OUTPUT_DIR=$EMB_DIR

echo "VQ test: model=$MODEL, M=$M"
echo "Codebook: $CODEBOOK"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/test_posthoc_prototypes.py \
    --model_dir $PRETRAINED/$MODEL \
    --method vq \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0

echo "Done."
