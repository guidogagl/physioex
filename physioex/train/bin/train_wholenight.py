"""Whole-night fine-tuning of the sequence level of a trained multimodal ChannelNet (xmai-lab/sleep-mechinterp#32).

The model of --init (run folder of a MultiChannelNet with F connections, e.g. MASS S3-F) is wrapped in
``WholeNightNet``: epoch level frozen, sequence level trained on long sequences (physioex/data/night_sequences.py).
Every configuration sees the same number of labelled epochs (--budget-passes x the training epochs); validation
(--n-vals times, whole validation nights in one pass, fused loss) selects best.pt, and the initial model competes.

Configurations (one change each, see the plan of #32):
    A-L20 A-mix A-whole A-crop A-perm A-noperm A-permrand A-noocc B-K4 B-K2 B-nods B-untied C C+A

Test (MASS fold 0): whole night in one pass and L 20 voting (fused); whole night per modality subset (occlusion) and
per auxiliary head. Per-night labels and predictions go to preds.npz for the paired statistics of analyse.py.

    python -m physioex.train.bin.train_wholenight --config A-mix --seed 0 --init <S3-F run> --root <cache v1> \
        --datasets mass_ss01 ... --split-file <json> --out <run folder>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from physioex.data.clean_cache import CleanCacheMultiNights
from physioex.data.night_sequences import MIXES, SequenceSampler, transition_matrix
from physioex.models.channelnet_mm import MultiChannelNet, occlusion_probs, sample_occlusion
from physioex.models.wholenight import WholeNightNet
from physioex.train.bin.train_channelnet import _metrics, git_sha
from physioex.train.bin.train_channelnet_mm import load_branches, to_dev, voting

#            mix        occlusion loops tied  deep_sup night  freeze_seq
CONFIGS = {
    "A-L20":      ("l20",      True,  1, True,  0.0, False, False),
    "A-mix":      ("mix",      True,  1, True,  0.0, False, False),
    "A-whole":    ("whole",    True,  1, True,  0.0, False, False),
    "A-crop":     ("crop",     True,  1, True,  0.0, False, False),
    "A-perm":     ("perm",     True,  1, True,  0.0, False, False),
    "A-noperm":   ("noperm",   True,  1, True,  0.0, False, False),
    "A-permrand": ("permrand", True,  1, True,  0.0, False, False),
    "A-noocc":    ("mix",      False, 1, True,  0.0, False, False),
    "B-K4":       ("mix",      True,  4, True,  0.3, False, False),
    "B-K2":       ("mix",      True,  2, True,  0.3, False, False),
    "B-nods":     ("mix",      True,  4, True,  0.0, False, False),
    "B-untied":   ("mix",      True,  4, False, 0.3, False, False),
    "C":          ("mix",      True,  1, True,  0.0, True,  True),
    "C+A":        ("mix",      True,  1, True,  0.0, True,  False),
}
CONFIG_LIST = list(CONFIGS)


def build_wholenight(init: Path, config: str):
    cfg = json.loads((init / "run.json").read_text())["config"]
    mods = cfg.get("modalities", ["EEG", "EOG", "EMG"])
    branches, kappa = load_branches(Path(cfg["branches"]), mods, 0.0, cfg.get("branch_prefix", ""))
    mm = MultiChannelNet(branches, cfg.get("points", []), hook=cfg.get("hook", "identity"))
    mm.load_state_dict(torch.load(init / "best.pt", map_location="cpu"))
    _, _, loops, tied, _, night, freeze = CONFIGS[config]
    return WholeNightNet(mm, loops=loops, tied=tied, night_head=night, freeze_seq=freeze), kappa, mods


def night_tensors(nt, device):
    x = {m: torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0).to(device) for m, v in nt.x.items()}
    off = {m: torch.full((1,), float(nt.offset_db[m]), device=device) for m in nt.x}
    return x, off


@torch.no_grad()
def whole_eval(model, nights, device, occlude=(), head="fused", crit=None):
    """Whole night in one forward pass (batch of one night). Returns (y_true, y_pred) per night and the mean loss."""
    model.eval()
    preds, losses = [], []
    for nt in nights:
        x, off = night_tensors(nt, device)
        occ = None if not occlude else {m: torch.tensor([m in occlude], device=device) for m in x}
        logits, aux, _ = model(x, off, occ)
        lg = (logits if head == "fused" else aux[head])[0]
        y = torch.from_numpy(nt.y.astype(np.int64)).to(device)
        if crit is not None:
            losses.append(float(crit(lg, y)))
        msk = nt.y >= 0
        preds.append((nt.y[msk], lg.argmax(-1).cpu().numpy()[msk]))
    return preds, (float(np.mean(losses)) if losses else None)


def pooled(preds):
    return _metrics(np.concatenate([a for a, _ in preds]), np.concatenate([b for _, b in preds]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=CONFIG_LIST)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", required=True, help="MultiChannelNet run folder (run.json + best.pt)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--pipe", default="702cb90947921189")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-passes", type=float, default=30.0, help="labelled epochs seen / training epochs")
    ap.add_argument("--n-vals", type=int, default=30)
    ap.add_argument("--lr-seq", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--aux", type=float, default=0.3)
    ap.add_argument("--occ-scale", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke runs: stop after this many steps")
    ap.add_argument("--no-train", action="store_true", help="parity check: evaluate the initial model only")
    ap.add_argument("--skip-test-extras", action="store_true")
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#34")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    cfg = {k: v for k, v in vars(args).items() if k != "out"}
    cfg["recipe"] = dict(zip(("mix", "occlusion", "loops", "tied", "deep_sup", "night_head", "freeze_seq"),
                             CONFIGS[args.config]))
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mix, use_occ, _, _, deep_sup, _, _ = CONFIGS[args.config]

    model, kappa, mods = build_wholenight(Path(args.init).expanduser(), args.config)
    model.to(device)
    split = json.loads(Path(args.split_file).read_text())
    data = CleanCacheMultiNights(args.root, args.datasets, mods, log=log, split=split, pipe=args.pipe)
    for k in data.nights:                                    # MASS is small: everything in memory
        for nt in data.nights[k]:
            nt.x = {m: np.ascontiguousarray(v) for m, v in nt.x.items()}
    train, valid, test = data.nights["train"], data.nights["valid"], data.nights["test"]
    trans = transition_matrix([nt.y for nt in train])
    probs = occlusion_probs(kappa, args.occ_scale) if use_occ else None
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)

    n_train_epochs = int(sum((nt.y >= 0).sum() for nt in train))
    budget = int(args.budget_passes * n_train_epochs)
    val_at = [int(budget * (i + 1) / args.n_vals) for i in range(args.n_vals)]
    groups = [{"params": ps, "lr": lr} for ps, lr in ((model.seq_parameters(), args.lr_seq),
                                                      (model.head_parameters(), args.lr_head)) if ps]
    opt = torch.optim.Adam(groups)

    def lr_factor(frac):
        w = args.warmup_frac
        return (frac + 1e-9) / w if frac < w else 0.5 * (1 + math.cos(math.pi * (frac - w) / (1 - w)))

    base_lr = [g["lr"] for g in opt.param_groups]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = {"issue": args.issue, "git_sha": git_sha(), "config": cfg, "config_hash": cfg_hash,
           "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
           "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "device": str(device), "n_trainable": n_trainable,
           "pretrained_kappa": kappa, "occlusion_probs": probs, "nights": {k: len(v) for k, v in data.nights.items()},
           "train_epochs": n_train_epochs, "budget_epochs": budget, "transition_matrix": trans.round(4).tolist(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log(f"{args.config} seed {args.seed}: recipe {cfg['recipe']}, {n_trainable} trainable, budget {budget} epochs "
        f"({args.budget_passes} x {n_train_epochs}), occlusion {probs}")

    _, v0 = whole_eval(model, valid, device, crit=crit)
    log(f"init valid loss {v0:.4f}")
    best, curve = v0, [(0, 0, None, v0)]
    torch.save(model.state_dict(), out / "best.pt")
    mf = open(out / "metrics.jsonl", "a")
    seen, step, vi, t0, run_loss = 0, 0, 0, time.time(), []
    sampler = iter(SequenceSampler(train, MIXES[mix], trans, seed=args.seed))
    while not args.no_train and seen < budget:
        b = next(sampler)
        x, off, y, lengths = to_dev(b["x"], device), to_dev(b["off"], device), b["y"].to(device), b["lengths"]
        B = y.shape[0]
        occ = None if probs is None else sample_occlusion(B, probs, mods, device)
        model.train()
        _, _, o = model(x, off, occ, lengths=None if mix == "l20" else lengths.to(device))

        def loop_loss(lg, aux):
            ls = crit(lg.reshape(-1, 5), y.reshape(-1))
            return ls + args.aux * sum(crit(aux[m].reshape(-1, 5), y.reshape(-1)) for m in mods)

        losses = [loop_loss(lg, aux) for lg, aux in o["loops"]]
        loss = losses[-1] + (deep_sup * sum(losses[:-1]) / (len(losses) - 1) if deep_sup and len(losses) > 1 else 0)
        f = lr_factor(seen / budget)
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        seen += int((y >= 0).sum())
        run_loss.append(float(loss))
        if not math.isfinite(run_loss[-1]):
            log("non-finite loss: run marked failed")
            run["failed"] = "non-finite loss"
            (out / "run.json").write_text(json.dumps(run, indent=2))
            raise SystemExit(3)
        while vi < len(val_at) and seen >= val_at[vi]:
            _, vl = whole_eval(model, valid, device, crit=crit)
            tl = float(np.mean(run_loss))
            run_loss = []
            vi += 1
            curve.append((step, seen, tl, vl))
            gains = {k: float(v.detach()) for k, v in model.gain.items()}
            if model.night is not None:
                gains["night"] = float(model.night.c.detach())
            mf.write(json.dumps({"step": step, "seen": seen, "train_loss": tl, "valid_loss": vl, "gains": gains,
                                 "min": round((time.time() - t0) / 60, 2)}) + "\n")
            mf.flush()
            log(f"val {vi}/{len(val_at)} step {step} seen {seen} train {tl:.4f} valid {vl:.4f} gains {gains}")
            if vl < best:
                best = vl
                torch.save(model.state_dict(), out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    preds = {}
    preds["whole"], _ = whole_eval(model, test, device)
    vote = voting(model, test, device, 20, return_preds=True)
    preds["vote20"] = [(a, b) for a, b in vote["preds"]]
    final = {"steps": step, "seen": seen, "minutes": round((time.time() - t0) / 60, 2), "valid_best_loss": best,
             "best_is_init": bool(best == v0), "curve": curve,
             "test": {"whole": pooled(preds["whole"]), "vote20": pooled(preds["vote20"])}}
    log(f"test κ whole {final['test']['whole']['kappa']:.4f} vote20 {final['test']['vote20']['kappa']:.4f}")
    if not args.skip_test_extras:
        for r in range(1, len(mods)):
            for keep in itertools.combinations(mods, r):
                k = "+".join(keep)
                preds[f"subset:{k}"], _ = whole_eval(model, test, device, occlude=set(mods) - set(keep))
                final["test"][f"subset:{k}"] = pooled(preds[f"subset:{k}"])
        for m in mods:
            preds[f"aux:{m}"], _ = whole_eval(model, test, device, head=m)
            final["test"][f"aux:{m}"] = pooled(preds[f"aux:{m}"])
        log("subsets/aux: " + " ".join(f"{k} {v['kappa']:.3f}" for k, v in final["test"].items()
                                       if ":" in k))
    np.savez_compressed(out / "preds.npz", nights=np.array([f"{nt.dataset}/{nt.subject}" for nt in test]),
                        **{f"{k}|{i}|{w}": arr for k, ps in preds.items() for i, (yt, yp) in enumerate(ps)
                           for w, arr in (("y", yt), ("p", yp))})
    run["final"] = final
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
