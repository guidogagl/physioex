#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=rand-seq
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%A_%a.out
#SBATCH --array=0-9

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy MASS cache to /dev/shm
echo "Copying MASS cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
for c in 01 02 03 04 05; do
    cp -r $CACHE_SRC/v1/mass_ss${c} /dev/shm/physioex_cache/v1/ 2>/dev/null
done
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

M_VALUES=(5 8 12 15 24 32 48 65 80 100)
M=${M_VALUES[$SLURM_ARRAY_TASK_ID]}

MODELS=/leonardo_work/IscrC_SELFEEG/pretrained/protosleepnet-seq-3ch-mixer
EMB=/leonardo_scratch/fast/IscrC_SELFEEG/epoch-embeddings/protosleepnet-seq-3ch-mixer
OUTDIR=/leonardo_work/IscrC_SELFEEG/results/randomization/seq
mkdir -p "$OUTDIR"

echo "Array task $SLURM_ARRAY_TASK_ID → M=$M"

python examples/pretrained/protosleepnet-gagliardi/test_prototype_randomization.py \
    --backbone seq \
    --checkpoint $MODELS/model.pt \
    --codebook_path $MODELS/vq_kmeans/${M}/codebook.npy \
    --emb_dir $EMB \
    --dataset mass --seq_len 20 --fold 0 --gpu_id 0 \
    --n_random 50 --seed 42 \
    --output_dir $OUTDIR
