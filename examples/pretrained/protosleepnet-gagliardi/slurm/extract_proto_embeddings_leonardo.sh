#!/bin/bash
# Launcher: submits one job per dataset for ProtoSleepNet embedding extraction.
# Usage: bash extract_proto_embeddings_leonardo.sh /path/to/model_dir
#
# NOT an sbatch script — run this from the login node.

MODEL_DIR=${1:?Usage: bash extract_proto_embeddings_leonardo.sh /path/to/model_dir}

SCRIPT=examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/extract_epoch_embeddings.py
OUTDIR=${MODEL_DIR}/posthoc_embeddings
CHANNELS="EEG EOG EMG"
CACHE=/leonardo_scratch/fast/IscrC_SELFEEG/cache/physioex

COMMON="source /leonardo_work/IscrC_SELFEEG/.venv/physioex-torch2x/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=$CACHE
cd /leonardo_work/IscrC_SELFEEG/code/physioex"

submit() {
    local name=$1
    shift
    sbatch --job-name="emb-${name}" \
        --partition=boost_usr_prod --gres=gpu:1 --ntasks-per-node=1 \
        --cpus-per-task=4 --mem=64G --account=IscrC_SELFEEG \
        --qos=boost_qos_lprod --time=02:00:00 \
        --output=/leonardo_work/IscrC_SELFEEG/logs/%x_%j.out \
        --wrap="$COMMON
python $SCRIPT --model_dir $MODEL_DIR --output_dir $OUTDIR --channels $CHANNELS --gpu_id 0 $*"
}

echo "Model: $MODEL_DIR"
echo "Output: $OUTDIR"
echo ""

# In-domain: MASS (train/valid/test)
submit "mass-indomain" --dataset mass

# Simple OOD datasets
for DS in sleepedf hmc dcsm mesa mros; do
    submit "$DS" --dataset $DS --dataset_name $DS
done

# SHHS
submit "shhs-v1" --dataset shhs --visit 1 --dataset_name shhs_visit1
submit "shhs-v2" --dataset shhs --visit 2 --dataset_name shhs_visit2

# MASS cohorts (OOD)
for C in 1 2 3 4 5; do
    submit "mass-c${C}" --dataset mass --cohort $C --dataset_name mass_cohort${C}
done

# HPAP
for SUBSET in lab-full lab-split; do
    submit "hpap-${SUBSET}" --dataset hpap --subset $SUBSET --dataset_name hpap_${SUBSET}
done

# WSC visits
for V in 1 2 3 4 5; do
    submit "wsc-v${V}" --dataset wsc --visit $V --dataset_name wsc_visit${V}
done

# Alzheimers
for SUBSET in AD HC; do
    submit "alz-${SUBSET}" --dataset alzheimers --subset $SUBSET --dataset_name alzheimers_${SUBSET}
done

# Parkinsons night
for GROUP in HOA PD; do
    submit "park-night-${GROUP}" --dataset parkinsons --recording night --group $GROUP --dataset_name parkinsons_night_${GROUP}
done

# Parkinsons nap
for GROUP in HOA PD; do
    submit "park-nap-${GROUP}" --dataset parkinsons --recording nap --group $GROUP --dataset_name parkinsons_nap_${GROUP}
done

echo ""
echo "All jobs submitted."
