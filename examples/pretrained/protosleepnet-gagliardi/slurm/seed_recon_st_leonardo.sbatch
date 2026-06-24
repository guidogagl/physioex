#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=08:00:00
#SBATCH --job-name=seed-recon-st
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

# Copy SHHS cache to /dev/shm
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

RESULTS=/leonardo_work/IscrC_SELFEEG/results/seed_stability/st
MODELS=/leonardo_work/IscrC_SELFEEG/pretrained/protosleepnet-st-3ch-mixer
LOGS=/leonardo_work/IscrC_SELFEEG/logs
RECON_SCRIPT=examples/pretrained/protosleepnet-gagliardi/proto-reconstruction/data_driven.py

echo "=== Seed reconstruction: data-driven, ST-SHHS, 7 seeds on 2 GPUs ==="
echo "=== Running 2 parallel streams (1 per GPU) to avoid SHHS I/O thrashing ==="

# GPU 0: seed_42 (production), 123, 789, 2048
(
    echo "  [GPU 0] seed 42 (production)..."
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python $RECON_SCRIPT \
        --backbone st --m 12 --top_k 256 --force \
        --checkpoint_path $MODELS/model.pt \
        --codebook_path $MODELS/posthoc_embeddings/protosleepnet-st-3ch-mixer/codebook_vq_kmeans_m12.npy \
        --output_dir $RESULTS/seed_42/data_driven \
        > ${LOGS}/seed-recon-st_${SLURM_JOB_ID}_s42.log 2>&1

    for S in 123 789 2048; do
        echo "  [GPU 0] seed $S..."
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        python $RECON_SCRIPT \
            --backbone st --m 12 --top_k 256 --force \
            --checkpoint_path $RESULTS/seed_$S/model.pt \
            --codebook_path $RESULTS/seed_$S/codebook_m12.npy \
            --output_dir $RESULTS/seed_$S/data_driven \
            > ${LOGS}/seed-recon-st_${SLURM_JOB_ID}_s${S}.log 2>&1
    done
    echo "  [GPU 0] done."
) &

# GPU 1: seed 456, 1024, 3141
(
    for S in 456 1024 3141; do
        echo "  [GPU 1] seed $S..."
        CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        python $RECON_SCRIPT \
            --backbone st --m 12 --top_k 256 --force \
            --checkpoint_path $RESULTS/seed_$S/model.pt \
            --codebook_path $RESULTS/seed_$S/codebook_m12.npy \
            --output_dir $RESULTS/seed_$S/data_driven \
            > ${LOGS}/seed-recon-st_${SLURM_JOB_ID}_s${S}.log 2>&1
    done
    echo "  [GPU 1] done."
) &

echo "Launched 2 GPU streams. Waiting..."
wait
echo "All seed reconstructions finished."
