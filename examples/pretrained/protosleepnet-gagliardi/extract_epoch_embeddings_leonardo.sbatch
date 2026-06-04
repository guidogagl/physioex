#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=06:00:00
#SBATCH --job-name=extract-emb
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# SHHS cache is 124 GB — too large for /dev/shm with 128G RAM.
# Read directly from FAST filesystem instead.
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
echo "Using cache at: $PHYSIOEX_CACHE_DIR"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
OUTPUT_DIR=/leonardo_work/IscrC_SELFEEG/pretrained/posthoc_embeddings
SCRIPT=examples/pretrained/protosleepnet-gagliardi/extract_epoch_embeddings.py

# Extract epoch embeddings from both baselines
echo ""
echo "================================================================"
echo "Extracting: st-baseline"
echo "================================================================"
python $SCRIPT \
    --model_dir $PRETRAINED/st-baseline \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0 \
    --batch_size 512

echo ""
echo "================================================================"
echo "Extracting: seq-baseline"
echo "================================================================"
python $SCRIPT \
    --model_dir $PRETRAINED/seq-baseline \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0 \
    --batch_size 512

echo ""
echo "================================================================"
echo "Extraction complete. Output:"
echo "================================================================"
ls -lh $OUTPUT_DIR/st-baseline/
ls -lh $OUTPUT_DIR/seq-baseline/
