#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=pilot_2026_0042
#SBATCH --time=06:00:00
#SBATCH --job-name=epoch-emb
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex-pip/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=/sofia/scratch/pilot/pilot_2026_0042/cache

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

MODEL_NAME="${MODEL_NAME:-sleeptransformer-phan}"
DS="${DS:-shhs}"
CHANNELS="${CHANNELS:-EEG}"

OUTPUT_DIR=/sofia/scratch/pilot/pilot_2026_0042/WORK/posthoc_embeddings

# Build extra args
EXTRA_ARGS=""
[ -n "$VISIT" ] && EXTRA_ARGS="$EXTRA_ARGS --visit $VISIT"
[ -n "$COHORT" ] && EXTRA_ARGS="$EXTRA_ARGS --cohort $COHORT"
[ -n "$SUBSET" ] && EXTRA_ARGS="$EXTRA_ARGS --subset $SUBSET"
[ -n "$DATASET_NAME" ] && EXTRA_ARGS="$EXTRA_ARGS --dataset_name $DATASET_NAME"

echo "Epoch embedding extraction: model=$MODEL_NAME, dataset=$DS, channels=$CHANNELS $EXTRA_ARGS"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/extract_epoch_embeddings.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --dataset $DS \
    --channels $CHANNELS \
    $EXTRA_ARGS \
    --gpu_id 0 \
    --batch_size 512

echo "Done."
