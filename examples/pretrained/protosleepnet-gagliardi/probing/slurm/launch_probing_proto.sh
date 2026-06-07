#!/bin/bash
# Launcher: submits linear probing sbatch jobs for protosleepnet-seq-3ch-mixer.
# Run from the Sofia login node AFTER cache_events jobs have completed.
# Usage: bash launch_probing_proto.sh

MODEL=protosleepnet-seq-3ch-mixer
EMB_ROOT=/sofia/scratch/pilot/pilot_2026_0042/WORK/posthoc_embeddings/$MODEL
OUTPUT_DIR=/sofia/scratch/pilot/pilot_2026_0042/WORK/probing/proto-seq-3ch-mixer
SBATCH=examples/pretrained/protosleepnet-gagliardi/probing/slurm/probe_sofia.sbatch

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

# ── Helper functions ───────────────────────────────────────────────

submit_probe() {
    # submit_probe SOURCE_NAME TIME CPUS EMB_DIRS...
    local src=$1 time=$2 cpus=$3
    shift 3
    # Build absolute paths from relative dirs
    local dirs=""
    for d in "$@"; do dirs="$dirs $EMB_ROOT/$d"; done
    sbatch --job-name="probe-${src}" --time=$time --cpus-per-task=$cpus \
        --export=ALL,SOURCE_NAME=$src,EMB_DIRS="$dirs",OUTPUT_DIR=$OUTPUT_DIR \
        $SBATCH
}

submit_discrim() {
    # submit_discrim SOURCE_NAME TASK TIME EMB_DIRS...
    local src=$1 task=$2 time=$3
    shift 3
    local dirs=""
    for d in "$@"; do dirs="$dirs $EMB_ROOT/$d"; done
    sbatch --job-name="probe-${src}" --time=$time --cpus-per-task=4 \
        --export=ALL,SOURCE_NAME=$src,TASK=$task,LABEL_FROM_DIR=1,EMB_DIRS="$dirs",OUTPUT_DIR=$OUTPUT_DIR \
        $SBATCH
}

echo "Submitting probing jobs for $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Standard probing (auto-discover tasks per dataset)
# ═══════════════════════════════════════════════════════════════════

# ── MASS in-domain ─────────────────────────────────────────────────
submit_probe mass_indomain 01:00:00 4  train valid test

# ── SHHS ───────────────────────────────────────────────────────────
submit_probe shhs_visit1   24:00:00 16 shhs_visit1/all
submit_probe shhs_visit2   12:00:00 16 shhs_visit2/all

# ── MESA / MrOS ───────────────────────────────────────────────────
submit_probe mesa          12:00:00 8  mesa/all
submit_probe mros          12:00:00 8  mros/all

# ── WSC visits ─────────────────────────────────────────────────────
submit_probe wsc_visit1    06:00:00 8  wsc_visit1/all
submit_probe wsc_visit2    04:00:00 8  wsc_visit2/all
submit_probe wsc_visit3    04:00:00 8  wsc_visit3/all
submit_probe wsc_visit4    01:00:00 4  wsc_visit4/all
submit_probe wsc_visit5    00:15:00 4  wsc_visit5/all

# ── MASS cohorts ───────────────────────────────────────────────────
for C in 1 2 3 4 5; do
    submit_probe mass_cohort${C} 00:30:00 4  mass_cohort${C}/all
done

# ── HPAP ───────────────────────────────────────────────────────────
submit_probe hpap_lab-full  01:00:00 4  hpap_lab-full/all
submit_probe hpap_lab-split 00:30:00 4  hpap_lab-split/all

# ── Sleep-EDF / HMC / DCSM ────────────────────────────────────────
submit_probe sleepedf      02:00:00 4  sleepedf/all
submit_probe hmc           01:00:00 4  hmc/all
submit_probe dcsm          02:00:00 4  dcsm/all

# ── Alzheimer's (AD + HC combined) ────────────────────────────────
submit_probe alzheimers    00:30:00 4  alzheimers_AD/all alzheimers_HC/all

# ── Parkinsons (HOA + PD combined per recording type) ─────────────
submit_probe parkinsons_night 00:30:00 4 parkinsons_night_HOA/all parkinsons_night_PD/all
submit_probe parkinsons_nap   00:15:00 4 parkinsons_nap_HOA/all parkinsons_nap_PD/all

# ═══════════════════════════════════════════════════════════════════
# Source discrimination (label_from_dir mode)
# ═══════════════════════════════════════════════════════════════════

submit_discrim shhs_visit  visit  02:00:00  shhs_visit1/all shhs_visit2/all

submit_discrim wsc_visit   visit  01:00:00  \
    wsc_visit1/all wsc_visit2/all wsc_visit3/all wsc_visit4/all wsc_visit5/all

submit_discrim mass_cohort cohort 00:30:00  \
    mass_cohort1/all mass_cohort2/all mass_cohort3/all mass_cohort4/all mass_cohort5/all

submit_discrim hpap_protocol protocol 00:15:00  hpap_lab-full/all hpap_lab-split/all

submit_discrim parkinsons_recording recording 00:15:00 \
    parkinsons_night_HOA/all parkinsons_night_PD/all \
    parkinsons_nap_HOA/all parkinsons_nap_PD/all

echo ""
echo "All probing jobs submitted."
