#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --job-name=vq-train
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd /leonardo_work/IscrC_SELFEEG/code/physioex

# Parameters (override via: sbatch --export=MODEL=seq-baseline,M=100 ...)
MODEL="${MODEL:-st-baseline}"
M="${M:-50}"
N_EPOCHS="${N_EPOCHS:-20}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL
OUTPUT_DIR=$EMB_DIR

echo "VQ training: model=$MODEL, M=$M, epochs=$N_EPOCHS"
echo "Embeddings: $EMB_DIR"

python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_vq.py \
    --model_dir $PRETRAINED/$MODEL \
    --emb_dir $EMB_DIR \
    --n_prototypes $M \
    --n_epochs $N_EPOCHS \
    --lr 1e-3 \
    --batch_size 2058 \
    --gpu_id 0 \
    --output_dir $OUTPUT_DIR

echo "Done. Output:"
ls -lh $OUTPUT_DIR/codebook_vq_m${M}*
