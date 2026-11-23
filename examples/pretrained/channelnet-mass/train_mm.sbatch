#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --job-name=channelnet-mm
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%A_%a.out
# Multimodal ChannelNet sweep on MASS SS01-SS05 (fold 0), one array task per connection setup.
# Env: CODE (checkout of feat/channelnet-mm), RUN (run folder), BRANCHES (folder with EEG/EOG/EMG best.pt + run.json),
#      PHASE = A  frozen branches: S1 late fusion, S2 with E, F, Q, F+E, F+E+Q          (array 0-5)
#              B  S3 fine-tuning from the PHASE A run of the same index                    (array 0-5, aftercorr:A)
#              C  setup $SETUP without occlusion: frozen stage, then S3 from it            (--array=0)
NAMES=(late E F Q FE FEQ)
POINTS=("" "E" "F" "Q" "F E" "F E Q")
case $PHASE in
  A) i=$SLURM_ARRAY_TASK_ID; NAME=${NAMES[$i]}; P=${POINTS[$i]}
     STAGE=$([ -z "$P" ] && echo S1 || echo S2); EXTRA=(); OUT=$RUN/A_$NAME ;;
  B) i=$SLURM_ARRAY_TASK_ID; NAME=${NAMES[$i]}; P=${POINTS[$i]}
     STAGE=S3; EXTRA=(--init $RUN/A_$NAME); OUT=$RUN/B_$NAME ;;
  C) for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = "$SETUP" ] && break; done; NAME=$SETUP; P=${POINTS[$i]}
     STAGE=S3; FIRST=$([ -z "$P" ] && echo S1 || echo S2)
     EXTRA=(--init $RUN/C_${NAME}_noocc_frozen --no-occlusion); OUT=$RUN/C_${NAME}_noocc ;;
  *) echo "PHASE must be A, B or C"; exit 1 ;;
esac
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
COMMON=(--branches $BRANCHES --root /sofia/projects/2026_101/cache/v1
  --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05
  --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json --num-workers 8
  --issue xmai-lab/sleep-mechinterp#19)
if [ "$PHASE" = C ]; then
  python -m physioex.train.bin.train_channelnet_mm --stage $FIRST ${P:+--points $P} --no-occlusion "${COMMON[@]}" \
    --out $RUN/C_${NAME}_noocc_frozen || exit 1
fi
python -m physioex.train.bin.train_channelnet_mm --stage $STAGE ${P:+--points $P} "${EXTRA[@]}" \
  "${COMMON[@]}" --out $OUT
