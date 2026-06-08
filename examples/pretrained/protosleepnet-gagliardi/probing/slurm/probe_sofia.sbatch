#!/bin/bash
#SBATCH --partition=zen5_dense
#SBATCH --cpus-per-task=4
#SBATCH --account=pilot_2026_0042
#SBATCH --time=06:00:00
#SBATCH --output=/sofia/scratch/pilot/pilot_2026_0042/WORK/logs/%x_%j.out

module swap cluster/sofia/zen4_h200 2>/dev/null
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load scikit-learn/1.7.0-gfbf-2025a
module load h5py/3.14.0-foss-2025a
module load PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0
source /sofia/scratch/pilot/pilot_2026_0042/WORK/.venv/physioex/bin/activate

export PYTHONUNBUFFERED=1

cd /sofia/scratch/pilot/pilot_2026_0042/WORK/physioex

SCRIPT=examples/pretrained/protosleepnet-gagliardi/probing/probe.py

# Environment variables set by launcher:
#   SOURCE_NAME  — probing source name (e.g. shhs_visit1, mesa)
#   EMB_DIRS     — space-separated absolute paths to embedding directories
#   OUTPUT_DIR   — probing output root
#   LABEL_FROM_DIR — if non-empty, source discrimination mode
#   TASK         — task name (required for source discrimination)
#   QUANTIZE     — (optional) path to codebook.npy for vector quantization

QUANTIZE_FLAG=""
[ -n "$QUANTIZE" ] && QUANTIZE_FLAG="--quantize $QUANTIZE"

echo "probe: source=$SOURCE_NAME"
echo "  emb_dirs=$EMB_DIRS"
echo "  output=$OUTPUT_DIR"
[ -n "$QUANTIZE" ] && echo "  quantize=$QUANTIZE"
echo ""

if [ -n "$LABEL_FROM_DIR" ]; then
    # ── Source discrimination mode ────────────────────────────────
    echo "=== Source discrimination: $TASK ==="
    if [ -f "$OUTPUT_DIR/$SOURCE_NAME/$TASK/summary.json" ]; then
        echo "  Already done, skipping"
    else
        python $SCRIPT --emb_dirs $EMB_DIRS --source_name $SOURCE_NAME \
            --task $TASK --label_from_dir $QUANTIZE_FLAG --output_dir $OUTPUT_DIR
    fi
else
    # ── Standard probing: discover tasks, iterate ─────────────────
    TASKS=$(python $SCRIPT --emb_dirs $EMB_DIRS --source_name $SOURCE_NAME \
            --output_dir $OUTPUT_DIR --discover 2>&1 | grep '^\s*\[' | awk '{print $2}')

    if [ -z "$TASKS" ]; then
        echo "ERROR: no tasks discovered"
        exit 1
    fi

    echo "Discovered tasks: $TASKS"
    echo ""

    for T in $TASKS; do
        if [ -f "$OUTPUT_DIR/$SOURCE_NAME/$T/summary.json" ]; then
            echo "=== $T === SKIP (already done)"
            continue
        fi
        echo "=== $T ==="
        python $SCRIPT --emb_dirs $EMB_DIRS --source_name $SOURCE_NAME \
            --task $T $QUANTIZE_FLAG --output_dir $OUTPUT_DIR
        echo ""
    done
fi

echo ""
echo "Done: $SOURCE_NAME"
