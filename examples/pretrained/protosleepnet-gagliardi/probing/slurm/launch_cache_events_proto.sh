#!/bin/bash
# Launcher: submits cache_events sbatch jobs for protosleepnet-seq-3ch-mixer embeddings.
# Run from the Sofia login node.
# Usage: bash launch_cache_events_proto.sh

SBATCH=examples/pretrained/protosleepnet-gagliardi/probing/slurm/cache_events_proto_sofia.sbatch
CVD_CSV=/sofia/scratch/pilot/pilot_2026_0042/raw_data/shhs/datasets/shhs-cvd-summary-dataset-0.21.0.csv

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

echo "Submitting cache_events jobs for protosleepnet-seq-3ch-mixer"
echo ""

# ── MASS in-domain (train/valid/test at root level) ────────────────
sbatch --job-name=cache-mass-indomain \
    --export=ALL,DS=mass,EMB_SUBDIRS="train valid test" \
    $SBATCH

# ── SHHS visit 1 (with CVD outcomes) ──────────────────────────────
sbatch --job-name=cache-shhs-v1 \
    --export=ALL,DS=shhs,VISIT=1,EMB_SUBDIRS="shhs_visit1/all",EXTRA_CSV=$CVD_CSV,EXTRA_CSV_KEY=nsrrid \
    $SBATCH

# ── SHHS visit 2 (with CVD outcomes) ──────────────────────────────
sbatch --job-name=cache-shhs-v2 \
    --export=ALL,DS=shhs,VISIT=2,EMB_SUBDIRS="shhs_visit2/all",EXTRA_CSV=$CVD_CSV,EXTRA_CSV_KEY=nsrrid \
    $SBATCH

# ── MESA ───────────────────────────────────────────────────────────
sbatch --job-name=cache-mesa \
    --export=ALL,DS=mesa,EMB_SUBDIRS="mesa/all" \
    $SBATCH

# ── MrOS ───────────────────────────────────────────────────────────
sbatch --job-name=cache-mros \
    --export=ALL,DS=mros,EMB_SUBDIRS="mros/all" \
    $SBATCH

# ── WSC visits ─────────────────────────────────────────────────────
for V in 1 2 3 4 5; do
    sbatch --job-name=cache-wsc-v${V} \
        --export=ALL,DS=wsc,VISIT=$V,EMB_SUBDIRS="wsc_visit${V}/all" \
        $SBATCH
done

# ── MASS cohorts ───────────────────────────────────────────────────
for C in 1 2 3 4 5; do
    sbatch --job-name=cache-mass-c${C} \
        --export=ALL,DS=mass,COHORT=$C,EMB_SUBDIRS="mass_cohort${C}/all" \
        $SBATCH
done

# ── HPAP ───────────────────────────────────────────────────────────
for SUBSET in lab-full lab-split; do
    sbatch --job-name=cache-hpap-${SUBSET} \
        --export=ALL,DS=hpap,SUBSET=$SUBSET,EMB_SUBDIRS="hpap_${SUBSET}/all" \
        $SBATCH
done

# ── Sleep-EDF ──────────────────────────────────────────────────────
sbatch --job-name=cache-sleepedf \
    --export=ALL,DS=sleepedf,EMB_SUBDIRS="sleepedf/all" \
    $SBATCH

# ── HMC ────────────────────────────────────────────────────────────
sbatch --job-name=cache-hmc \
    --export=ALL,DS=hmc,EMB_SUBDIRS="hmc/all" \
    $SBATCH

# ── DCSM ───────────────────────────────────────────────────────────
sbatch --job-name=cache-dcsm \
    --export=ALL,DS=dcsm,EMB_SUBDIRS="dcsm/all" \
    $SBATCH

# ── Alzheimer's (AD and HC separately) ────────────────────────────
sbatch --job-name=cache-alz-AD \
    --export=ALL,DS=alzheimers,SUBSET=AD,EMB_SUBDIRS="alzheimers_AD/all" \
    $SBATCH

sbatch --job-name=cache-alz-HC \
    --export=ALL,DS=alzheimers,SUBSET=HC,EMB_SUBDIRS="alzheimers_HC/all" \
    $SBATCH

# ── Parkinsons night ──────────────────────────────────────────────
for GROUP in HOA PD; do
    sbatch --job-name=cache-park-night-${GROUP} \
        --export=ALL,DS=parkinsons,RECORDING=night,GROUP=$GROUP,EMB_SUBDIRS="parkinsons_night_${GROUP}/all" \
        $SBATCH
done

# ── Parkinsons nap ────────────────────────────────────────────────
for GROUP in HOA PD; do
    sbatch --job-name=cache-park-nap-${GROUP} \
        --export=ALL,DS=parkinsons,RECORDING=nap,GROUP=$GROUP,EMB_SUBDIRS="parkinsons_nap_${GROUP}/all" \
        $SBATCH
done

echo ""
echo "All cache_events jobs submitted."
