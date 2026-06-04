#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --job-name=nmf-proto
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1

cd /leonardo_work/IscrC_SELFEEG/code/physioex

MODEL="${MODEL:-st-baseline}"
K="${K:-10}"

PRETRAINED=/leonardo_work/IscrC_SELFEEG/pretrained
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL

echo "NMF prototype discovery: model=$MODEL, k_per_class=$K"

python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_nmf.py \
    --emb_dir $EMB_DIR \
    --k_per_class $K \
    --max_iter 500

echo "Done."
ls -lh $EMB_DIR/prototypes_nmf_k${K}*
