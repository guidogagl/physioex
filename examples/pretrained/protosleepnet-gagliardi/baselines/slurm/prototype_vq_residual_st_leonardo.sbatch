#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --job-name=vq-stres
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

# Usage: sbatch --export=MODEL_DIR=/path/to/model,BUILD_MODULE=train_st_1ch_residual prototype_vq_residual_st_leonardo.sbatch

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy SHHS cache to /dev/shm
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
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
        --dataset shhs --channels EEG --gpu_id 0
else
    echo "=== Embeddings already extracted, skipping ==="
fi

for M in 5 8 12 15 24 32 48 65 80 100; do
    echo ""
    echo "============================================================"
    echo "M = $M"
    echo "============================================================"

    # 1. K-Means only
    KMEANS_CB="$EMB_DIR/codebook_vq_kmeans_m${M}.npy"
    if [ ! -f "$KMEANS_CB" ]; then
        echo "--- K-Means ---"
        python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
            --emb_dir "$EMB_DIR" --n_prototypes $M
    fi

    echo "--- Test K-Means ---"
    python $SCRIPTS/test_residual_prototypes.py \
        --build_module "$BUILD_MODULE" --checkpoint "$CKPT" \
        --codebook_path "$KMEANS_CB" \
        --output_dir "$MODEL_DIR/vq_kmeans_results" \
        --dataset shhs --channels EEG --seq_len 21 --gpu_id 0

    # 2. K-Means init + supervised refinement
    echo "--- VQ K-Means+supervised ---"
    python $SCRIPTS/learn_residual_prototypes_vq.py \
        --build_module "$BUILD_MODULE" --checkpoint "$CKPT" \
        --emb_dir "$EMB_DIR" --n_prototypes $M \
        --output_dir "$EMB_DIR" --gpu_id 0 --seq_len 21

    VQ_CB="$EMB_DIR/codebook_vq_m${M}.npy"
    echo "--- Test VQ K-Means+supervised ---"
    python $SCRIPTS/test_residual_prototypes.py \
        --build_module "$BUILD_MODULE" --checkpoint "$CKPT" \
        --codebook_path "$VQ_CB" \
        --output_dir "$MODEL_DIR/vq_supervised_results" \
        --dataset shhs --channels EEG --seq_len 21 --gpu_id 0

    # 3. Learned (no K-Means init) + supervised refinement
    echo "--- VQ learned ---"
    python $SCRIPTS/learn_residual_prototypes_vq.py \
        --build_module "$BUILD_MODULE" --checkpoint "$CKPT" \
        --emb_dir "$EMB_DIR" --n_prototypes $M \
        --output_dir "$EMB_DIR" --gpu_id 0 --seq_len 21 --no_kmeans_init

    LEARNED_CB="$EMB_DIR/codebook_vq_learned_m${M}.npy"
    echo "--- Test VQ learned ---"
    python $SCRIPTS/test_residual_prototypes.py \
        --build_module "$BUILD_MODULE" --checkpoint "$CKPT" \
        --codebook_path "$LEARNED_CB" \
        --output_dir "$MODEL_DIR/vq_learned_results" \
        --dataset shhs --channels EEG --seq_len 21 --gpu_id 0
done

echo ""
echo "=== All done ==="
echo "Results in $MODEL_DIR/{vq_kmeans_results,vq_supervised_results,vq_learned_results}/"
