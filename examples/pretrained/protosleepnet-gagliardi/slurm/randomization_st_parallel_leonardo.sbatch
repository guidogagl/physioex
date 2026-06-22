#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=rand-st-par
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%A_%a.out
#SBATCH --array=0-9

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

# Copy SHHS cache to /dev/shm (shared by all workers, ~124 GB)
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

M_VALUES=(5 8 12 15 24 32 48 65 80 100)
M=${M_VALUES[$SLURM_ARRAY_TASK_ID]}

MODELS=/leonardo_work/IscrC_SELFEEG/pretrained/protosleepnet-st-3ch-mixer
POSTHOC=$MODELS/posthoc_embeddings/protosleepnet-st-3ch-mixer
OUTDIR=/leonardo_work/IscrC_SELFEEG/results/randomization/st
mkdir -p "$OUTDIR"

echo "Array task $SLURM_ARRAY_TASK_ID → M=$M (ST backbone, SHHS, parallel)"

# 2 GPUs, 4 workers per GPU = 8 workers total
# Each worker: 2 CPU cores, ~300MB GPU, ~4GB RAM
N_PER_GPU=4
N_GPUS=2
N_WORKERS=$((N_PER_GPU * N_GPUS))

for w in $(seq 0 $((N_WORKERS-1))); do
    GPU_ID=$((w / N_PER_GPU))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    python examples/pretrained/protosleepnet-gagliardi/test_prototype_randomization.py \
        --backbone st \
        --checkpoint $MODELS/model.pt \
        --codebook_path $POSTHOC/codebook_vq_kmeans_m${M}.npy \
        --emb_dir $POSTHOC \
        --dataset shhs --seq_len 21 --fold 0 --gpu_id 0 \
        --n_random 20 --seed 42 \
        --worker_id $w --n_workers $N_WORKERS \
        --output_dir $OUTDIR \
        > /leonardo_work/IscrC_SELFEEG/logs/rand-st-par_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_w${w}.log 2>&1 &
done

echo "Launched $N_WORKERS workers for M=$M. Waiting..."
wait
echo "All workers finished for M=$M"
