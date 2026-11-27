#!/bin/bash
# Submit members of the model zoo (xmai-lab/sleep-mechinterp#38), one job each (examples/pretrained/zoo/member.sbatch).
#   submit.sh <group> [TRAIN] [SEED]      group: z0 | stageA | name "epoch-seq" (several allowed, comma-separated)
# Env: CODE (zoo checkout on Sofia), ROOT (stage folder), optional ISSUE, optional DEP (sbatch --dependency value).
set -eu
GROUP=$1; TRAIN=${2:-mass}; SEED=${3:-42}
Z0=(lstm-gru)
EAXIS=(tf-gru patch-gru rawcnn-gru s4d-gru mamba-gru mlstm-gru slstm-gru)
SAXIS=(lstm-tf lstm-s4d lstm-mamba lstm-mlstm lstm-slstm lstm-tcn)
HOMOG=(tf-tf s4d-s4d mamba-mamba mlstm-mlstm rawcnn-tcn)
case $GROUP in
  z0) MEMBERS=("${Z0[@]}") ;;
  stageA) MEMBERS=("${EAXIS[@]}" "${SAXIS[@]}" "${HOMOG[@]}") ;;
  *) IFS=, read -ra MEMBERS <<< "$GROUP" ;;
esac
mkdir -p $ROOT
for m in "${MEMBERS[@]}"; do
  E=${m%-*}; S=${m#*-}
  sbatch --parsable --job-name=zoo-$TRAIN-$m-s$SEED ${DEP:+--dependency=$DEP} \
    --export=ALL,CODE=$CODE,ROOT=$ROOT,EPOCH=$E,SEQ=$S,SEED=$SEED,TRAIN=$TRAIN,ISSUE=${ISSUE:-xmai-lab/sleep-mechinterp#41} \
    $CODE/examples/pretrained/zoo/member.sbatch | sed "s/^/$m /"
done
