#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_bprod
#SBATCH --time=04:00:00
#SBATCH --job-name=extract-emb
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy SHHS cache to /dev/shm for fast I/O
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

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
