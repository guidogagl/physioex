#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=seed-seq
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1

# Copy MASS cache to /dev/shm (shared by all workers)
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
LOGS=/leonardo_work/IscrC_SELFEEG/logs
mkdir -p "$RESULTS"

echo "=== Seed stability: SEQ backbone, MASS, 6 seeds on 2 GPUs ==="

# 6 seeds, 3 per GPU
SEEDS=(123 456 789 1024 2048 3141)
N_PER_GPU=3

for i in "${!SEEDS[@]}"; do
    S=${SEEDS[$i]}
    GPU_ID=$((i / N_PER_GPU))
    OUTDIR=$RESULTS/seed_$S
    mkdir -p "$OUTDIR"

    echo "  Launching seed $S on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    python examples/pretrained/protosleepnet-gagliardi/seed_stability_pipeline.py \
        --backbone seq --dataset mass --seed $S \
        --output_dir $OUTDIR --gpu_id 0 --n_prototypes 12 \
        --num_workers 0 \
        > ${LOGS}/seed-seq_${SLURM_JOB_ID}_s${S}.log 2>&1 &
done

echo "Launched ${#SEEDS[@]} workers. Waiting..."
wait
echo "All seeds finished."
