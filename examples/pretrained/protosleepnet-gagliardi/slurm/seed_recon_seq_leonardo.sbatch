#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=seed-recon
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

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

RESULTS=/leonardo_work/IscrC_SELFEEG/results/seed_stability/seq
MODELS=/leonardo_work/IscrC_SELFEEG/pretrained/protosleepnet-seq-3ch-mixer
LOGS=/leonardo_work/IscrC_SELFEEG/logs
RECON_SCRIPT=examples/pretrained/protosleepnet-gagliardi/proto-reconstruction/data_driven.py

echo "=== Seed reconstruction: data-driven, SEQ, 7 seeds on 2 GPUs ==="

# Seed 42 (production model — different paths)
echo "  Launching seed 42 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python $RECON_SCRIPT \
    --backbone seq --m 12 --top_k 256 --force \
    --checkpoint_path $MODELS/model.pt \
    --codebook_path $MODELS/embeddings/codebook_vq_kmeans_m12.npy \
    --output_dir $RESULTS/seed_42/data_driven \
    > ${LOGS}/seed-recon_${SLURM_JOB_ID}_s42.log 2>&1 &

# Seeds 123-3141 (seed stability models)
SEEDS=(123 456 789 1024 2048 3141)
for i in "${!SEEDS[@]}"; do
    S=${SEEDS[$i]}
    GPU_ID=$(( (i + 1) % 2 ))  # distribute: 123→1, 456→0, 789→1, 1024→0, 2048→1, 3141→0

    echo "  Launching seed $S on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python $RECON_SCRIPT \
        --backbone seq --m 12 --top_k 256 --force \
        --checkpoint_path $RESULTS/seed_$S/model.pt \
        --codebook_path $RESULTS/seed_$S/codebook_m12.npy \
        --output_dir $RESULTS/seed_$S/data_driven \
        > ${LOGS}/seed-recon_${SLURM_JOB_ID}_s${S}.log 2>&1 &
done

echo "Launched 7 workers. Waiting..."
wait
echo "All seed reconstructions finished."
