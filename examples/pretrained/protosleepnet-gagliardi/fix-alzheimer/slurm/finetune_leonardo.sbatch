#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=finetune-alz
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

# Usage: sbatch --export=BACKBONE=seq finetune_leonardo.sbatch
#        sbatch --export=BACKBONE=st,LR=1e-3 finetune_leonardo.sbatch

BACKBONE=${BACKBONE:-seq}
LR=${LR:-1e-5}
FROM_SCRATCH=${FROM_SCRATCH:-}

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex

MODELS_DIR=/leonardo_work/IscrC_SELFEEG/pretrained
if [ -n "$FROM_SCRATCH" ]; then
    OUTDIR=${MODELS_DIR}/finetune-alzheimer-${BACKBONE}-lr${LR}-scratch
else
    OUTDIR=${MODELS_DIR}/finetune-alzheimer-${BACKBONE}-lr${LR}
fi
mkdir -p "$OUTDIR"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

EXTRA_ARGS=""
if [ -n "$FROM_SCRATCH" ]; then
    EXTRA_ARGS="--from_scratch"
else
    if [ "$BACKBONE" = "seq" ]; then
        CKPT=${MODELS_DIR}/protosleepnet-seq-3ch-mixer/model.pt
    else
        CKPT=${MODELS_DIR}/protosleepnet-st-3ch-mixer/model.pt
    fi
    EXTRA_ARGS="--checkpoint $CKPT"
fi

python examples/pretrained/protosleepnet-gagliardi/fix-alzheimer/finetune_alzheimer.py \
    --backbone "$BACKBONE" \
    $EXTRA_ARGS \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --lr $LR \
    --max_epochs 50 \
    --patience 10 \
    --batch_size 32
