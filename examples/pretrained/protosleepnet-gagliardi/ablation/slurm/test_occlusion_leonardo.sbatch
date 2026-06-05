#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --job-name=occ-test
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy ONLY shhs_visit1 cache to /dev/shm
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd /leonardo_work/IscrC_SELFEEG/code/physioex

MODEL="${MODEL:-st-baseline}"
MODEL_DIR="/leonardo_work/IscrC_SELFEEG/pretrained/$MODEL"
echo "Testing occlusion robustness for: $MODEL"

python examples/pretrained/protosleepnet-gagliardi/ablation/test_occlusion.py \
    --model_dir "$MODEL_DIR" \
    --gpu_id 0 \
    --num_workers 14
