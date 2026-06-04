#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=ood-emb
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex

cd /leonardo_work/IscrC_SELFEEG/code/physioex

# Parameters (override via --export)
MODEL="${MODEL:-st-baseline}"
DS="${DS:-hmc}"
DS_KWARGS="${DS_KWARGS:-{}}"
DS_NAME="${DS_NAME:-$DS}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
OUTPUT_DIR=$PRETRAINED/posthoc_embeddings

echo "OOD embedding extraction: model=$MODEL, dataset=$DS_NAME"

python examples/pretrained/protosleepnet-gagliardi/extract_epoch_embeddings.py \
    --model_dir $PRETRAINED/$MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DS \
    --dataset_kwargs "$DS_KWARGS" \
    --dataset_name $DS_NAME \
    --gpu_id 0 \
    --batch_size 512

echo "Done."
