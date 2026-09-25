"""One run of the xLSTM-in-SeqSleepNet study: (model, dataset, fold, seed) -> metrics + per-subject predictions.

Protocols (selected with --dataset):
    mass      Phan et al. 2019 — MASS SS1-SS5 combined (200 subjects), 20-fold subject CV,
              180 train / 10 valid / 10 test per fold (--n_folds 20 --n_valid 10).
    sleepedf  Phan et al. 2023 — Sleep-EDF SC 2013 subset (20 subjects, 39 nights, Fpz-Cz),
              leave-one-subject-out CV with 4 validation subjects, both nights of a subject
              kept together (--n_folds 20 --n_valid 4 --subset 2013).
    shhs      Phan et al. 2023 — SHHS visit 1, fixed 70/30 benchmark split (fold 0, --n_folds 0).

Models (--model):
    seqsleepnet    physioex.models.seqsleepnet:SeqSleepNet          (--model_kwargs JSON)
    lseqsleepnet   physioex.models.lseqsleepnet:LSeqSleepNet        (paper-compliant; --model_kwargs JSON)
    xseqsleepnet   physioex.models.xseqsleepnet:XSeqSleepNet        (--epoch_encoder, --sequence_encoder,
                                                                     --epoch_kwargs / --seq_kwargs JSON)

Outputs in --out_dir/<run_name>/:
    config.json                  everything needed to reproduce the run
    metrics_<mode>.json          pooled metrics + per-subject aggregated stats (bootstrap CI)
    predictions_<mode>.pt        {"subject_ids", "logits", "targets"} for paired comparisons
    checkpoints/                 Trainer checkpoints

Examples:
    # Stage 0a: SeqSleepNet parity, MASS 3-channel, fold 0
    python run.py --dataset mass --channels EEG EOG EMG --model seqsleepnet --L 20 \
        --n_folds 20 --n_valid 10 --fold 0 --seed 0 --lr 1e-4 --batch_size 32 --max_epochs 10

    # Stage 0b: L-SeqSleepNet gate on SleepEDF (LOSO fold 3)
    python run.py --dataset sleepedf --subset 2013 --channels EEG --model lseqsleepnet --L 200 \
        --n_folds 20 --n_valid 4 --fold 3 --lr 1e-4 --weight_decay 1e-4 --adam_eps 1e-7 --batch_size 8

    # Stage 2: flat xLSTM (bidirectional) at L=200 vs GRU controls
    python run.py --dataset mass --model xseqsleepnet --sequence_encoder xlstm_bi \
        --seq_kwargs '{"num_blocks": 2, "num_heads": 4}' --L 200 --fold 0 --seed 0 --batch_size 8

    # Stage 4: causal xLSTM, evaluated both with window voting and one pass over the night
    python run.py --dataset mass --model xseqsleepnet --sequence_encoder xlstm_causal --L 200 \
        --eval_modes voting single_pass ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from physioex.data.datasets import get_dataset
from physioex.data.multi import MultiDataset
from physioex.data.splits import assign_kfold_splits
from physioex.train.trainer import Trainer, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / protocol
    p.add_argument("--dataset", choices=["mass", "sleepedf", "shhs"], required=True)
    p.add_argument("--dataset_root", default=None, help="dataset root (else PHYSIOEX_DATA conventions)")
    p.add_argument("--cohorts", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="MASS cohorts")
    p.add_argument("--subset", default=None, help="dataset subset (sleepedf: '2013' = 20-subject SC set)")
    p.add_argument("--channels", nargs="+", default=["EEG"])
    p.add_argument("--L", type=int, default=20, help="training sequence length (epochs)")
    p.add_argument("--n_folds", type=int, default=20, help="0 = dataset's own get_splits (e.g. SHHS benchmark)")
    p.add_argument("--n_valid", type=int, default=10, help="validation subject groups per fold")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=42)
    # model
    p.add_argument("--model", choices=["seqsleepnet", "lseqsleepnet", "xseqsleepnet"], required=True)
    p.add_argument("--model_kwargs", type=json.loads, default={}, help="JSON kwargs (seqsleepnet/lseqsleepnet)")
    p.add_argument("--epoch_encoder", default="seqsleepnet")
    p.add_argument("--sequence_encoder", default="gru")
    p.add_argument("--epoch_kwargs", type=json.loads, default={})
    p.add_argument("--seq_kwargs", type=json.loads, default={})
    # optimisation
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (validation rounds)")
    p.add_argument("--valid_interval_ratio", type=float, default=0.1)
    p.add_argument("--accumulate", type=int, default=1)
    # evaluation
    p.add_argument("--eval_modes", nargs="+", default=["voting"], choices=["voting", "single_pass"])
    p.add_argument("--n_bootstrap", type=int, default=1000)
    # infra
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="outputs/xlstm-seqsleepnet")
    p.add_argument("--run_name", default=None)
    p.add_argument("--smoke", action="store_true", help="1 training epoch, tiny validation, for pipeline checks")
    p.add_argument("--eval_only", default=None, metavar="MODEL_PT",
                   help="skip training: load this state_dict and only run the evaluation modes")
    return p.parse_args()


def build_dataset(args):
    """Return (dataset, group_fn) with the protocol's k-fold split installed."""
    kw = dict(channels=args.channels, pipelines="seqsleepnet", sequence_length=args.L)
    if args.dataset_root:
        kw["root"] = args.dataset_root
    group_fn = None

    if args.dataset == "mass":
        MASS = get_dataset("mass")
        parts = []
        for c in args.cohorts:
            ds = MASS(cohort=c, **kw)
            n = ds.get_n_subjects()
            print(f"  MASS SS{c:02d}: {n} subjects")
            if n > 0:
                parts.append(ds)
        members = parts
        dataset = MultiDataset(parts) if len(parts) > 1 else parts[0]
    elif args.dataset == "sleepedf":
        if args.subset:
            kw["subset"] = args.subset
        dataset = get_dataset("sleepedf")(**kw)
        members = [dataset]
        group_fn = dataset._subject_group_key  # both nights of a subject together
        print(f"  SleepEDF: {dataset.get_n_subjects()} recordings")
    else:
        dataset = get_dataset("shhs")(visit=1, **kw)
        members = [dataset]
        print(f"  SHHS v1: {dataset.get_n_subjects()} subjects")

    if args.n_folds > 0:
        assign_kfold_splits(members, n_folds=args.n_folds, n_valid=args.n_valid,
                            seed=args.split_seed, group_fn=group_fn)
    return dataset


def build_model(args, in_chan: int):
    if args.model == "seqsleepnet":
        from physioex.models.seqsleepnet import SeqSleepNet

        return SeqSleepNet(in_chan=in_chan, **args.model_kwargs), "physioex.models.seqsleepnet:SeqSleepNet"
    if args.model == "lseqsleepnet":
        from physioex.models.lseqsleepnet import LSeqSleepNet

        m = LSeqSleepNet(in_chan=in_chan, **args.model_kwargs)
        if m.sequence_length != args.L:
            raise SystemExit(f"--L must equal B*K={m.sequence_length} for L-SeqSleepNet")
        return m, "physioex.models.lseqsleepnet:LSeqSleepNet"
    from physioex.models.xseqsleepnet import XSeqSleepNet

    m = XSeqSleepNet(in_chan=in_chan, epoch_encoder=args.epoch_encoder,
                     sequence_encoder=args.sequence_encoder,
                     epoch_kwargs=args.epoch_kwargs, seq_kwargs=args.seq_kwargs)
    return m, "physioex.models.xseqsleepnet:XSeqSleepNet"


def _jsonable(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def main():
    args = parse_args()
    if args.smoke:
        args.max_epochs, args.valid_interval_ratio = 1, 0.5

    tag = args.model if args.model != "xseqsleepnet" else f"x_{args.epoch_encoder}_{args.sequence_encoder}"
    run_name = args.run_name or f"{args.dataset}_{tag}_{len(args.channels)}ch_L{args.L}_f{args.fold}_s{args.seed}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    print(f"[run] {run_name}")
    dataset = build_dataset(args)
    model, model_class = build_model(args, in_chan=len(args.channels))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_class} params={n_params:,}")

    config = {
        "run_name": run_name, "argv": sys.argv[1:], "args": vars(args),
        "model_class": model_class, "n_params": n_params,
        "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(_jsonable(config), indent=2))

    def _load_weights(path):
        obj = torch.load(path, map_location="cpu")
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        model.load_state_dict(state)
        return obj.get("epoch") if isinstance(obj, dict) else None

    t0 = time.time()
    if args.eval_only:
        ep = _load_weights(args.eval_only)
        print(f"[eval_only] loaded {args.eval_only} (epoch={ep})")
        train_seconds = 0.0
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     eps=args.adam_eps)
        model = Trainer.train(
            model=model, dataset=dataset, optimizer=optimizer, lr=args.lr, weight_decay=args.weight_decay,
            max_epochs=args.max_epochs, train_batch_size=args.batch_size, fold=args.fold,
            gpu_id=args.gpu_id, checkpoint_path=str(out / "checkpoints"),
            early_stopping_patience=args.patience, valid_interval_ratio=args.valid_interval_ratio,
            accumulate_grad_batches=args.accumulate, num_workers=args.num_workers,
            pin_memory=args.num_workers > 0, persistent_workers=args.num_workers > 0, prefetch_factor=2,
        )
        train_seconds = time.time() - t0
        # Trainer.train returns the last-epoch weights; the protocol (Phan) retains the
        # model that performed best on the validation set, which Trainer saved as the
        # single checkpoint under checkpoints/. Restore it before evaluating.
        ckpts = sorted((out / "checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            ep = _load_weights(ckpts[-1])
            print(f"[best] restored {ckpts[-1].name} (epoch={ep})")
        else:
            print("[best] WARNING: no checkpoint found, evaluating last-epoch weights")
        torch.save(model.cpu().state_dict(), out / "model.pt")

    summary = {"train_seconds": train_seconds}
    for mode in args.eval_modes:
        t1 = time.time()
        res = Trainer.voting_evaluate(
            model=model, dataset=dataset, L=args.L, fold=args.fold, gpu_id=args.gpu_id,
            per_subject=True, n_bootstrap=args.n_bootstrap, mode=mode, return_predictions=True,
            seed=args.seed,
        )
        preds = res.pop("predictions")
        res["eval_seconds"] = time.time() - t1
        torch.save(preds, out / f"predictions_{mode}.pt")
        (out / f"metrics_{mode}.json").write_text(json.dumps(_jsonable(res), indent=2))
        summary[mode] = {k: res[k] for k in ("accuracy", "f1_score", "cohen_kappa") if k in res}
        print(f"[eval:{mode}] acc={res['accuracy']:.4f} mf1={res['f1_score']:.4f} "
              f"kappa={res['cohen_kappa']:.4f} ({res['eval_seconds']:.0f}s, "
              f"{len(preds['subject_ids'])} subjects)")

    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    print(f"[done] {run_name} train={train_seconds/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
