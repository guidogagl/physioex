#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --account=zen4-h200-2026_101-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=/sofia/projects/2026_101/outputs/sleep-mechinterp/logs/%x_%j.out
# One member of the model zoo of the multimodal framework (xmai-lab/sleep-mechinterp#38): the whole framework in one
# submission, every step skipped when its run.json already holds "final" (resubmit the same command after a timeout).
#   1. branches EEG, EOG, EMG (ZooChannel EPOCH/SEQ), three processes in parallel on the GPU
#        mass: protocol phan, MASS SS01-05 fold 0 (as #16)    shhs: protocol st-phan, SHHS-1 split (as #28)
#   2. fusion: branches frozen, F connections on the 4 epoch blocks + fused head (S2, --points F)
#   3. joint fine-tuning from step 2 (S3); steps 2-3 with the arguments of #19 B_F (mass) / #29 (shhs)
#   4. evaluation of the final model (eval_ood.py, voting protocol, at most 300 nights per cohort):
#        in-domain test split, the 12 clinical/pediatric evaluation-only cohorts, the population cohorts of the other
#        training set
# Env: CODE (checkout of the zoo branch), ROOT (stage folder), EPOCH, SEQ, SEED, TRAIN (mass|shhs).
set -u
source /sofia/projects/2026_101/WORK/.venv/physioex-pip/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=$CODE
cd $CODE
CACHE=/sofia/projects/2026_101/cache/v1
NAME=${EPOCH}-${SEQ}-s${SEED}
RUN=$ROOT/$NAME
mkdir -p $RUN
EVAL_ONLY=(alzheimers_ad alzheimers_hc ccshs dcsm dreams hmc hpap lsec parkinsons_nap_hoa parkinsons_nap_pd
           parkinsons_night_hoa parkinsons_night_pd)
done_() { [ -f "$1/run.json" ] && grep -q '"final"' "$1/run.json"; }
log() { echo "[$(date +%F' '%T)] $NAME: $*"; }

if [ "$TRAIN" = mass ]; then
  DATA=(--root $CACHE --datasets mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05
        --split-file examples/pretrained/channelnet-mass/mass_fold0_split.json)
  BR_ARGS=(--protocol phan --num-workers 6)
  MM_ARGS=(--num-workers 8)
  S2_ARGS=(); S3_ARGS=()
  INDOMAIN=(mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05)
  SPLIT=examples/pretrained/channelnet-mass/mass_fold0_split.json
  POP=(mesa mros shhs_visit2 wsc_visit1 wsc_visit2 wsc_visit3 wsc_visit4)
  POP_SPLIT=(shhs_visit1)            # SHHS-1: test part of its split only, comparable with the SHHS-trained members
  POP_SPLIT_FILE=examples/pretrained/st-shhs/shhs1_split.json
else
  LOCAL=/tmp/$USER-$SLURM_JOB_ID; trap 'rm -rf $LOCAL' EXIT
  DATA=(--root $CACHE --datasets shhs_visit1 --split-file examples/pretrained/st-shhs/shhs1_split.json)
  BR_ARGS=(--protocol st-phan --workers 6 --num-workers 2)
  MM_ARGS=(--local-dir $LOCAL/mm --workers 20 --num-workers 4 --seq-len 21 --batch 32 --eps 1e-7 --val-every 100
           --patience 200 --min-vals 5000 --select loss)
  S2_ARGS=(--sched const --steps 1000000); S3_ARGS=(--steps 500000 --warmup-frac 0.1)
  INDOMAIN=(shhs_visit1)
  SPLIT=examples/pretrained/st-shhs/shhs1_split.json
  POP=(mass_ss01 mass_ss02 mass_ss03 mass_ss04 mass_ss05 mesa mros wsc_visit1 wsc_visit2 wsc_visit3 wsc_visit4)
  POP_SPLIT=()
fi
ISSUE=${ISSUE:-xmai-lab/sleep-mechinterp#41}

# 1. branches -----------------------------------------------------------------------------------------------------
pids=()
for M in EEG EOG EMG; do
  if done_ $RUN/$M; then log "branch $M done"; continue; fi
  EXTRA=()
  [ "$TRAIN" = shhs ] && EXTRA=(--local-dir $LOCAL/$M)
  python -m physioex.train.bin.train_channelnet --arch zoo --epoch-enc $EPOCH --seq-enc $SEQ --modality $M \
    --seed $SEED "${DATA[@]}" "${BR_ARGS[@]}" "${EXTRA[@]}" --issue $ISSUE --out $RUN/$M > $RUN/train_$M.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
for M in EEG EOG EMG; do done_ $RUN/$M || { log "branch $M FAILED (see $RUN/train_$M.log)"; exit 1; }; done
log "branches done"

# 2-3. fusion, then joint fine-tuning -----------------------------------------------------------------------------
MM="python -m physioex.train.bin.train_channelnet_mm --branches $RUN --points F --seed $SEED ${DATA[*]} ${MM_ARGS[*]}
    --issue $ISSUE"
done_ $RUN/S2_F || $MM --stage S2 "${S2_ARGS[@]}" --out $RUN/S2_F || { log "S2 FAILED"; exit 1; }
done_ $RUN/S3_F || $MM --stage S3 --init $RUN/S2_F "${S3_ARGS[@]}" --out $RUN/S3_F || { log "S3 FAILED"; exit 1; }
log "framework done"

# 4. evaluation ---------------------------------------------------------------------------------------------------
EV="python examples/pretrained/channelnet-mass/eval_ood.py --models $NAME=$RUN/S3_F --root $CACHE --max-nights 300
    --issue $ISSUE"
mkdir -p $RUN/eval
[ -f $RUN/eval/indomain/.done ] || { $EV --datasets ${INDOMAIN[*]} --split-file $SPLIT --split-part test \
    --max-nights 0 --out $RUN/eval/indomain && touch $RUN/eval/indomain/.done; }
[ -f $RUN/eval/eval_only/.done ] || { $EV --datasets ${EVAL_ONLY[*]} --out $RUN/eval/eval_only \
    && touch $RUN/eval/eval_only/.done; }
[ -f $RUN/eval/population/.done ] || { $EV --datasets ${POP[*]} --out $RUN/eval/population \
    && { [ ${#POP_SPLIT[@]} -eq 0 ] || $EV --datasets ${POP_SPLIT[*]} --split-file $POP_SPLIT_FILE --split-part test \
         --out $RUN/eval/population; } && touch $RUN/eval/population/.done; }
for g in indomain eval_only population; do [ -f $RUN/eval/$g/.done ] || { log "eval $g FAILED"; exit 1; }; done
log "DONE"
