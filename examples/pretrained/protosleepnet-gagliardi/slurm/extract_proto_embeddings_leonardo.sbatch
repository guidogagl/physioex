#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=IscrC_SELFEEG
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=12:00:00
#SBATCH --job-name=proto-extract
#SBATCH --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out

# Extracts epoch embeddings from ProtoSleepNet for all datasets (in-domain + OOD)
# Usage: sbatch --export=MODEL_DIR=/path/to/model extract_proto_embeddings_leonardo.sbatch

source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex

cd /leonardo_work/IscrC_SELFEEG/code/physioex

SCRIPT=examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/extract_epoch_embeddings.py
OUTDIR=${MODEL_DIR}/posthoc_embeddings
CHANNELS="EEG EOG EMG"

echo "Model: $MODEL_DIR"
echo "Output: $OUTDIR"

# ── In-domain: MASS (train/valid/test) ─────────────────────────────
echo ""
echo "========== IN-DOMAIN: MASS =========="
python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
    --dataset mass --channels $CHANNELS --gpu_id 0

# ── OOD datasets ───────────────────────────────────────────────────

# Simple datasets (no extra params)
for DS in sleepedf hmc dcsm mesa mros; do
    echo ""
    echo "========== OOD: $DS =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset $DS --dataset_name $DS --channels $CHANNELS --gpu_id 0
done

# SHHS visit 1 (OOD for SEQ backbone)
echo ""
echo "========== OOD: shhs_visit1 =========="
python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
    --dataset shhs --visit 1 --dataset_name shhs_visit1 --channels $CHANNELS --gpu_id 0

# SHHS visit 2
echo ""
echo "========== OOD: shhs_visit2 =========="
python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
    --dataset shhs --visit 2 --dataset_name shhs_visit2 --channels $CHANNELS --gpu_id 0

# MASS cohorts (OOD for ST backbone, but useful for all)
for C in 1 2 3 4 5; do
    echo ""
    echo "========== OOD: mass_cohort${C} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset mass --cohort $C --dataset_name mass_cohort${C} --channels $CHANNELS --gpu_id 0
done

# HPAP
for SUBSET in lab-full lab-split; do
    echo ""
    echo "========== OOD: hpap_${SUBSET} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset hpap --subset $SUBSET --dataset_name hpap_${SUBSET} --channels $CHANNELS --gpu_id 0
done

# WSC visits
for V in 1 2 3 4 5; do
    echo ""
    echo "========== OOD: wsc_visit${V} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset wsc --visit $V --dataset_name wsc_visit${V} --channels $CHANNELS --gpu_id 0
done

# Alzheimers
for SUBSET in AD HC; do
    echo ""
    echo "========== OOD: alzheimers_${SUBSET} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset alzheimers --subset $SUBSET --dataset_name alzheimers_${SUBSET} --channels $CHANNELS --gpu_id 0
done

# Parkinsons night
for GROUP in HOA PD; do
    echo ""
    echo "========== OOD: parkinsons_night_${GROUP} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset parkinsons --recording night --group $GROUP --dataset_name parkinsons_night_${GROUP} --channels $CHANNELS --gpu_id 0
done

# Parkinsons nap
for GROUP in HOA PD; do
    echo ""
    echo "========== OOD: parkinsons_nap_${GROUP} =========="
    python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR \
        --dataset parkinsons --recording nap --group $GROUP --dataset_name parkinsons_nap_${GROUP} --channels $CHANNELS --gpu_id 0
done

echo ""
echo "========== ALL DONE =========="
echo "Output: $OUTDIR"
ls -d $OUTDIR/*/
