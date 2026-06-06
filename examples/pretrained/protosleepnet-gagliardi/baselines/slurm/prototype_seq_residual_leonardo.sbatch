#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=proto-res
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

# Usage: sbatch --export=MODEL_DIR=/path/to/model,BUILD_MODULE=train_seq_1ch_residual prototype_seq_residual_leonardo.sbatch

module load profile/deeplrn
module load cineca-ai/4.0.0
source /leonardo_work/IscrC_SELFEEG/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy MASS cache to /dev/shm
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
for c in 01 02 03 04 05; do
    cp -r $CACHE_SRC/v1/mass_ss${c} /dev/shm/physioex_cache/v1/ 2>/dev/null
done
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex
SCRIPTS=examples/pretrained/protosleepnet-gagliardi/baselines

# Find checkpoint
CKPT=$(ls "$MODEL_DIR/checkpoints/"epoch=*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT="$MODEL_DIR/model.pt"
fi
echo "Checkpoint: $CKPT"
echo "Build module: $BUILD_MODULE"

EMB_DIR="$MODEL_DIR/embeddings"

# Step 1: Extract embeddings (skip if already done)
if [ ! -d "$EMB_DIR/train" ]; then
    echo "=== Extracting embeddings ==="
    python $SCRIPTS/extract_residual_embeddings.py \
        --build_module "$BUILD_MODULE" \
        --checkpoint "$CKPT" \
        --output_dir "$EMB_DIR" \
        --dataset mass --channels EEG --gpu_id 0
else
    echo "=== Embeddings already extracted, skipping ==="
fi

# Step 2+3: For each M, learn K-Means codebook + test
KMEANS_SCRIPT=examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py

for M in 5 8 12 15 24 32 48 65 80 100; do
    echo ""
    echo "=============================="
    echo "M = $M"
    echo "=============================="

    CODEBOOK="$EMB_DIR/codebook_vq_kmeans_m${M}.npy"

    # Learn codebook (CPU, fast)
    if [ ! -f "$CODEBOOK" ]; then
        python $KMEANS_SCRIPT --emb_dir "$EMB_DIR" --n_prototypes $M
    else
        echo "Codebook M=$M already exists, skipping"
    fi

    # Test with VQ prototypes
    python $SCRIPTS/test_residual_prototypes.py \
        --build_module "$BUILD_MODULE" \
        --checkpoint "$CKPT" \
        --codebook_path "$CODEBOOK" \
        --output_dir "$MODEL_DIR/vq_results" \
        --dataset mass --channels EEG --seq_len 20 --gpu_id 0
done

echo ""
echo "=== All done ==="
echo "Results in $MODEL_DIR/vq_results/"
