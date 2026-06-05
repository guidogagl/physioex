#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --job-name=vq-probe
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd /leonardo_work/IscrC_SELFEEG/code/physioex

MODEL="${MODEL:-st-baseline}"
DS_NAME="${DS_NAME:-hmc}"
M="${M:-48}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL/$DS_NAME/all
CODEBOOK=$PRETRAINED/posthoc_embeddings/$MODEL/codebook_vq_m${M}.npy
OUTPUT_DIR=$PRETRAINED/posthoc_embeddings/$MODEL/probe_vq_m${M}/$DS_NAME

echo "VQ probe: model=$MODEL, dataset=$DS_NAME, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/probe_posthoc_prototypes.py \
    --model_dir $PRETRAINED/$MODEL \
    --codebook_path $CODEBOOK \
    --emb_dir $EMB_DIR \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0

echo "Done."
