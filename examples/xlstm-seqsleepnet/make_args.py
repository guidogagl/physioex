"""Generate the run.py argument files for every stage of the xLSTM-in-SeqSleepNet study.

The files are consumed by run.sbatch through RUN_ARGS_FILE (they may contain JSON with
commas, which `sbatch --export` would otherwise split). ``{fold}`` and ``{seed}`` are
placeholders that run.sbatch replaces with the array task id and $SEED.

    python make_args.py --out <outputs>/_args --mass_root <...>/MASS/Original --sleepedf_root <...>/physionet-sleep-data

Grids (as executed on Sofia, Sep 2026):
    stage0a   SeqSleepNet (physioex), MASS 3ch, 20-fold Phan, L=20            -> s0a_mass_seqsleepnet.txt
    stage0a2  XSeqSleepNet epoch="paper", GRU 1 layer, λ 1e-3                  -> s0a_paper.txt
    stage0b   L-SeqSleepNet, SleepEDF-2013 LOSO, L=200                        -> s0b_sedf_lseqsleepnet.txt
    stage1    L=20, stride 1, batch 32: gru | gru_wrapped | gru_matched | xlstm_bi
    stage2    L in {100,200,400}, stride L/20, batch 8: gru | gru_wrapped | gru_matched | xlstm_bi | lseqsleepnet
    sweep     fold 0, L=200: xlstm_bi lr {3e-4,1e-3} x blocks {2,4} x heads {4,8}; controls x lr {3e-4,1e-3}
"""
from __future__ import annotations

import argparse
from pathlib import Path

XK = '{"num_blocks":2,"num_heads":4}'
ARMS = {
    "gru": "--model seqsleepnet",
    "gru_wrapped": '--model xseqsleepnet --sequence_encoder gru_wrapped --seq_kwargs {"num_blocks":2}',
    "gru_matched": f"--model xseqsleepnet --sequence_encoder gru_matched --seq_kwargs {XK}",
    "xlstm_bi": f"--model xseqsleepnet --sequence_encoder xlstm_bi --seq_kwargs {XK}",
}


def lseq(L: int) -> str:
    return f'--model lseqsleepnet --model_kwargs {{"B":10,"K":{L // 10}}} --weight_decay 1e-4 --adam_eps 1e-7'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass_root", required=True)
    ap.add_argument("--sleepedf_root", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mass = (f"--dataset mass --dataset_root {args.mass_root} --channels EEG EOG EMG --n_folds 20 --n_valid 10 "
            f"--lr 1e-4 --max_epochs 10 --patience 10 --eval_modes voting --fold {{fold}}")
    files = {}

    # Stage 0
    files["s0a_mass_seqsleepnet"] = f"{mass} --model seqsleepnet --L 20 --batch_size 32 --seed {{seed}}"
    files["s0a_paper"] = (f"{mass} --model xseqsleepnet --epoch_encoder paper "
                          '--epoch_kwargs {"hidden_size":64,"attention_size":64,"dropout":0.25,"recurrent_bn":true} '
                          '--sequence_encoder gru --seq_kwargs {"hidden":64,"num_layers":1} '
                          "--L 20 --batch_size 32 --weight_decay 1e-3 --seed {seed}")
    files["s0b_sedf_lseqsleepnet"] = (
        f"--dataset sleepedf --subset 2013 --dataset_root {args.sleepedf_root} --channels EEG --model lseqsleepnet "
        "--L 200 --n_folds 20 --n_valid 4 --lr 1e-4 --weight_decay 1e-4 --adam_eps 1e-7 --batch_size 8 "
        "--max_epochs 10 --patience 50 --valid_interval_ratio 0.027 --eval_modes voting --fold {fold} --seed {seed}")

    # Stage 1
    for arm, spec in ARMS.items():
        files[f"s1_{arm}_L20"] = f"{mass} {spec} --L 20 --batch_size 32 --seed {{seed}}"

    # Stage 2
    for L in (100, 200, 400):
        common = f"{mass} --L {L} --train_stride {L // 20} --batch_size 8 --seed {{seed}}"
        for arm, spec in ARMS.items():
            files[f"s2_{arm}_L{L}"] = f"{common} {spec}"
        files[f"s2_lseqsleepnet_L{L}"] = f"{common} {lseq(L)}"

    # Fairness sweep (fold 0, seed 0, L=200) -- lr overrides the --lr 1e-4 in `mass` (argparse keeps the last)
    base = f"{mass} --L 200 --train_stride 10 --batch_size 8 --seed 0".replace("{fold}", "0")
    for lr in ("3e-4", "1e-3"):
        for nb in (2, 4):
            for nh in (4, 8):
                name = f"sweep_xlstm_bi_L200_lr{lr}_b{nb}_h{nh}"
                files[name] = (f"{base} --model xseqsleepnet --sequence_encoder xlstm_bi "
                               f'--seq_kwargs {{"num_blocks":{nb},"num_heads":{nh}}} --lr {lr} --run_name {name}_f0_s0')
        for arm, spec in ARMS.items():
            name = f"sweep_{arm}_L200_lr{lr}"
            files[name] = f"{base} {spec} --lr {lr} --run_name {name}_f0_s0"
        name = f"sweep_lseqsleepnet_L200_lr{lr}"
        files[name] = f"{base} {lseq(200)} --lr {lr} --run_name {name}_f0_s0"

    # v2: Stage 1+2 re-run with the validation-selected learning rate per arm (fold-0 sweep, L=200):
    # 1e-3 for every flat arm, 3e-4 for L-SeqSleepNet. --tag keeps the outputs apart from the lr-1e-4 runs.
    LR_FLAT, LR_LSEQ = "1e-3", "3e-4"
    for arm, spec in ARMS.items():
        files[f"v2_s1_{arm}_L20"] = f"{mass} {spec} --L 20 --batch_size 32 --lr {LR_FLAT} --tag lr{LR_FLAT} --seed {{seed}}"
    for L in (100, 200, 400):
        common = f"{mass} --L {L} --train_stride {L // 20} --batch_size 8 --seed {{seed}}"
        for arm, spec in ARMS.items():
            files[f"v2_s2_{arm}_L{L}"] = f"{common} {spec} --lr {LR_FLAT} --tag lr{LR_FLAT}"
        files[f"v2_s2_lseqsleepnet_L{L}"] = f"{common} {lseq(L)} --lr {LR_LSEQ} --tag lr{LR_LSEQ}"

    # Stage 0a at the selected lr: 20-fold parity claim for the baseline
    files["s0a_mass_seqsleepnet_lr1e-3"] = (f"{mass} --model seqsleepnet --L 20 --batch_size 32 --lr {LR_FLAT} "
                                            f"--tag lr{LR_FLAT} --seed {{seed}}")

    # Stage 4: causal single-pass inference. Causal mLSTM vs unidirectional GRU (both at the selected lr),
    # evaluated with window voting AND one forward pass over the whole night; train at L=20 too, to test
    # length extrapolation (train short, infer on the full night).
    CAUSAL = {
        "xlstm_causal": f"--model xseqsleepnet --sequence_encoder xlstm_causal --seq_kwargs {XK}",
        "gru_uni": '--model xseqsleepnet --sequence_encoder gru_uni --seq_kwargs {"hidden":128,"num_layers":4}',
    }
    for L in (20, 200):
        stride = 1 if L == 20 else L // 20
        bs = 32 if L == 20 else 8
        for arm, spec in CAUSAL.items():
            files[f"s4_{arm}_L{L}"] = (f"{mass} {spec} --L {L} --train_stride {stride} --batch_size {bs} --lr {LR_FLAT} "
                                       f"--tag lr{LR_FLAT} --eval_modes voting single_pass --seed {{seed}}")

    for name, content in files.items():
        (out / f"{name}.txt").write_text(content + "\n")
    print(f"wrote {len(files)} argument files to {out}")


if __name__ == "__main__":
    main()
