#!/bin/bash
#SBATCH --partition=zen5_dense
#SBATCH --cpus-per-task=4
#SBATCH --account=pilot_2026_0042
#SBATCH --time=06:00:00
#SBATCH --job-name=cache-labels
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

module swap cluster/sofia/zen4_h200 2>/dev/null
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load h5py/3.14.0-foss-2025a
module load PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0
source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1
export PHYSIOEX_DATA=/sofia/scratch/pilot/pilot_2026_0042/raw_data
export PHYSIOEX_CACHE_DIR=/sofia/scratch/pilot/pilot_2026_0042/cache

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

DS="${DS:-shhs}"
CHANNELS="${CHANNELS:-EEG}"
EMB_ROOT=/sofia/scratch/pilot/pilot_2026_0042/WORK/posthoc_embeddings

# Build extra args from environment
EXTRA=""
[ -n "$VISIT" ] && EXTRA="$EXTRA --visit $VISIT"
[ -n "$COHORT" ] && EXTRA="$EXTRA --cohort $COHORT"
[ -n "$SUBSET" ] && EXTRA="$EXTRA --subset $SUBSET"
[ -n "$RECORDING" ] && EXTRA="$EXTRA --recording $RECORDING"
[ -n "$GROUP" ] && EXTRA="$EXTRA --group $GROUP"
[ -n "$EXTRA_CSV" ] && EXTRA="$EXTRA --extra_csv $EXTRA_CSV"
[ -n "$EXTRA_CSV_KEY" ] && EXTRA="$EXTRA --extra_csv_key $EXTRA_CSV_KEY"
[ -n "$EXTRA_CSV_FILTER" ] && EXTRA="$EXTRA --extra_csv_filter $EXTRA_CSV_FILTER"

SCRIPT=examples/pretrained/protosleepnet-gagliardi/probing/cache_events.py

echo "cache_events: dataset=$DS channels=$CHANNELS $EXTRA"
echo "EMB_SUBDIRS=$EMB_SUBDIRS"

for MODEL in sleeptransformer-phan seqsleepnet-phan; do
    for EMB_SUBDIR in $EMB_SUBDIRS; do
        DIR=$EMB_ROOT/$MODEL/$EMB_SUBDIR
        if [ ! -d "$DIR" ]; then
            echo "[SKIP] $DIR does not exist"
            continue
        fi
        echo ""
        echo "=== $MODEL / $EMB_SUBDIR ==="
        python $SCRIPT --dataset $DS --channels $CHANNELS $EXTRA --emb_dir $DIR
    done
done

echo ""
echo "All done."
