#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=seed-pred
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
LOGS=/leonardo_work/IscrC_SELFEEG/logs

echo "=== Convert checkpoints + predict: SEQ, 6 seeds on 2 GPUs ==="

SEEDS=(123 456 789 1024 2048 3141)

for i in "${!SEEDS[@]}"; do
    S=${SEEDS[$i]}
    GPU_ID=$((i / 3))

    echo "  Launching seed $S on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    python examples/pretrained/protosleepnet-gagliardi/convert_and_predict_seeds.py \
        --seed_dir $RESULTS/seed_$S \
        --backbone seq --dataset mass --seq_len 20 --gpu_id 0 \
        > ${LOGS}/seed-pred_${SLURM_JOB_ID}_s${S}.log 2>&1 &
done

echo "Launched ${#SEEDS[@]} workers. Waiting..."
wait
echo "All seeds finished."
