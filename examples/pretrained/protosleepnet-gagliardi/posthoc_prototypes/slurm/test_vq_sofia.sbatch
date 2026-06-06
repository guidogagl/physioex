#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=pilot_2026_0042
#SBATCH --time=03:00:00
#SBATCH --job-name=vq-test
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex-pip/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy dataset cache to /dev/shm if SHM_DATASET is set
CACHE_SRC=/sofia/scratch/pilot/pilot_2026_0042/cache
SHM=/dev/shm/physioex_cache
if [ -n "$SHM_DATASET" ] && [ -d "$CACHE_SRC/v1/$SHM_DATASET" ]; then
    echo "Copying $SHM_DATASET cache to /dev/shm..."
    mkdir -p $SHM/v1
    cp -r $CACHE_SRC/v1/$SHM_DATASET $SHM/v1/
    export PHYSIOEX_CACHE_DIR=$SHM
    echo "Cache copied: $(du -sh $SHM | cut -f1)"
else
    export PHYSIOEX_CACHE_DIR=$CACHE_SRC
fi

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

MODEL_NAME="${MODEL_NAME:-seqsleepnet-phan}"
M="${M:-48}"
CODEBOOK_TYPE="${CODEBOOK_TYPE:-vq_learned}"
DS="${DS:-mass}"
SEQ_LEN="${SEQ_LEN:-20}"
CHANNELS="${CHANNELS:-EEG}"

EMB_DIR="/sofia/scratch/pilot/pilot_2026_0042/WORK/posthoc_embeddings/${MODEL_NAME}"
CODEBOOK="${EMB_DIR}/codebook_${CODEBOOK_TYPE}_m${M}.npy"
OUTPUT_DIR="${EMB_DIR}"

echo "VQ test: model=$MODEL_NAME, M=$M, codebook=$CODEBOOK_TYPE, dataset=$DS"

# Build extra args
EXTRA_ARGS=""
[ -n "$COHORT" ] && EXTRA_ARGS="$EXTRA_ARGS --cohort $COHORT"
[ -n "$VISIT" ] && EXTRA_ARGS="$EXTRA_ARGS --visit $VISIT"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/test_posthoc_prototypes.py \
    --model_name $MODEL_NAME \
    --method vq \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0 \
    --dataset $DS \
    --channels $CHANNELS \
    --seq_len $SEQ_LEN \
    $EXTRA_ARGS

echo "Done."
