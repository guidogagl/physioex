"""Train the multimodal ChannelNet (physioex/models/channelnet_mm.py) from three pretrained single-channel ChannelNets.

Stages (seqsleepnet-phan protocol otherwise: L = 20, batch 32, Adam, 10 passes, stride 1, validation every 10 % of a
pass, best checkpoint by validation loss of the fused head, voting test):
    S1  late fusion: branches frozen, no connections, fused head only (lr 1e-3)
    S2  connections: branches frozen, gates + W + fused head (lr 1e-3)
    S3  fine-tuning from an S1/S2 checkpoint: new parameters lr 1e-4, branches lr 1e-5, 1-pass linear warmup then
        cosine decay; optional L2-SP penalty towards the pretrained branch weights (--l2sp)

Branches: ChannelNet or SleepTransformerChannel (read from each branch run.json; xmai-lab/sleep-mechinterp#26).
S0 trains a 3-channel baseline from scratch: --arch ssn3ch (SeqSleepNet) or st3ch (SleepTransformer).
SleepTransformer paper protocol: --seq-len 21 --eps 1e-7 --val-every 100 --patience 200 --min-vals 5000.

Occlusion (on unless --no-occlusion): per sample, modality m hidden with p_m = 0.5 kappa_m / max kappa, kappa from the
pretrained branches' run.json. Loss = CE(fused) + lambda * sum_m CE(aux_m).

Evaluation on the test split (voting): all modalities; every subset of modalities (occlusion at test); each connection
point switched off; each branch alone (all connections off, its own head) against its pretrained kappa; gate values.

    python -m physioex.train.bin.train_channelnet_mm --stage S2 --points F E Q --branches <dir with EEG/EOG/EMG> \
        --root <cache> --datasets mass_ss01 ... --split-file <json> --out <run folder>
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

from physioex.data.clean_cache import (CleanCacheMultiNights, MultiRandomOffsetSequences, MultiSequenceDataset,
                                       multi_class_counts)
from physioex.models.channelnet_mm import MODALITIES, MultiChannelNet, occlusion_probs, sample_occlusion
from physioex.models.seqsleepnet_3ch_clean import SeqSleepNet3chClean
from physioex.models.seqsleepnet_clean import ChannelNet
from physioex.models.sleeptransformer_clean import SleepTransformer3chClean, SleepTransformerChannel
from physioex.models.zoo_channel import ZooChannel
from physioex.train.bin.train_channelnet import _metrics, git_sha

STAGE_LR = {"S0": (1e-4, None), "S1": (1e-3, None), "S2": (1e-3, None), "S3": (1e-4, 1e-5)}   # (new params, branches)


def to_dev(d, device):
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


def load_branches(bdir: Path, mods, dropout: float = 0.0, prefix: str = ""):
    """Pretrained branches <bdir>/<prefix><m>/{best.pt, run.json}; the backbone is read from each run.json
    (ChannelNet, SleepTransformerChannel or ZooChannel, the last two with the dropout they were trained with)."""
    branches, kappa = {}, {}
    for m in mods:
        bdir_m = bdir / f"{prefix}{m}"
        run = json.loads((bdir_m / "run.json").read_text())
        cfg = run.get("config", {})
        if cfg.get("arch") == "zoo":
            net = ZooChannel(m, cfg.get("epoch_enc", "lstm"), cfg.get("seq_enc", "gru"),
                             dropout=cfg.get("resolved", {}).get("dropout", 0.0))
        elif cfg.get("arch", "channelnet") == "sleeptransformer":
            net = SleepTransformerChannel(m, dropout=cfg.get("resolved", {}).get("dropout", 0.1))
        else:
            net = ChannelNet(m, dropout=dropout)
        net.load_state_dict(torch.load(bdir_m / "best.pt", map_location="cpu"))
        branches[m] = net
        kappa[m] = float(run["final"]["test"]["kappa"])
    return branches, kappa


@torch.no_grad()
def voting(model, nights, device, L, occlude=None, disabled=(), head="fused", bs=64, return_preds=False):
    """Trainer.voting_evaluate protocol: L offsets, logits averaged per epoch. occlude: set of hidden modalities;
    disabled: connection sites switched off; head: 'fused' or a modality (its auxiliary head).
    return_preds: also res["preds"] = [(y_true, y_pred) per evaluated night]."""
    model.eval()
    model.disabled = set(disabled)
    yt_all, yp_all, per_ds = [], [], {}
    for nt in nights:
        n = len(nt.y)
        if n < L:
            continue
        xs = {m: torch.from_numpy(np.asarray(v, dtype=np.float32)) for m, v in nt.x.items()}
        votes = torch.zeros(n, 5, device=device)
        counts = torch.zeros(n, device=device)
        for o in range(L):
            usable = (n - o) - (n - o) % L
            if usable == 0:
                continue
            xw = {m: v[o:o + usable].reshape(-1, L, v.shape[-1]) for m, v in xs.items()}
            nw = next(iter(xw.values())).shape[0]
            for i in range(0, nw, bs):
                xb = {m: v[i:i + bs].to(device) for m, v in xw.items()}
                b = next(iter(xb.values())).shape[0]
                off = {m: torch.full((b,), nt.offset_db[m], device=device) for m in xb}
                occ = None if not occlude else {m: torch.full((b,), m in occlude, device=device) for m in xb}
                logits, aux, _ = model(xb, off, occ)
                lg = logits if head == "fused" else aux[head]
                a = o + i * L
                votes[a:a + b * L] += lg.reshape(-1, 5).float()
                counts[a:a + b * L] += 1
        yp = (votes / counts.clamp(min=1).unsqueeze(-1)).argmax(-1).cpu().numpy()
        msk = nt.y >= 0
        yt_all.append(nt.y[msk])
        yp_all.append(yp[msk])
        per_ds.setdefault(nt.dataset, []).append(len(yt_all) - 1)
    model.disabled = set()
    res = _metrics(np.concatenate(yt_all), np.concatenate(yp_all))
    res["per_dataset"] = {}
    for ds, idx in sorted(per_ds.items()):
        yt = np.concatenate([yt_all[i] for i in idx])
        yp = np.concatenate([yp_all[i] for i in idx])
        res["per_dataset"][ds] = {"kappa": float(cohen_kappa_score(yt, yp)), "nights": len(idx)}
    if return_preds:
        res["preds"] = list(zip(yt_all, yp_all))
    return res


@torch.no_grad()
def tiled_eval(model, nights, device, L, crit, bs=64):
    """Validation: tiled windows, fused loss and kappa, plus each branch alone (connections off, own head)."""
    model.eval()
    ds = MultiSequenceDataset(nights, L, stride=L)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    has_aux = isinstance(model, MultiChannelNet)
    losses, yt, yp, alone = [], [], [], {m: [] for m in model.mods} if has_aux else {}
    for x, y, off in dl:
        x, off, y = to_dev(x, device), to_dev(off, device), y.to(device)
        logits, _, _ = model(x, off)
        losses.append(float(crit(logits.reshape(-1, 5), y.reshape(-1))))
        aux = {}
        if has_aux:
            model.disabled = set(model.sites)
            _, aux, _ = model(x, off)
            model.disabled = set()
        msk = (y >= 0).reshape(-1).cpu().numpy()
        yt.append(y.reshape(-1).cpu().numpy()[msk])
        yp.append(logits.argmax(-1).reshape(-1).cpu().numpy()[msk])
        for m in alone:
            alone[m].append(aux[m].argmax(-1).reshape(-1).cpu().numpy()[msk])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {"loss": float(np.mean(losses)), "kappa": float(cohen_kappa_score(yt, yp)),
            "acc": float(accuracy_score(yt, yp)), "macro_f1": float(f1_score(yt, yp, average="macro")),
            "alone_kappa": {m: float(cohen_kappa_score(yt, np.concatenate(v))) for m, v in alone.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mm", choices=["mm", "ssn3ch", "st3ch"],
                    help="mm: MultiChannelNet from pretrained branches; ssn3ch: 3-channel SeqSleepNet baseline (S0)")
    ap.add_argument("--stage", required=True, choices=["S0", "S1", "S2", "S3"], help="S0: train from scratch")
    ap.add_argument("--points", nargs="*", default=[], choices=["F", "E", "Q"])
    ap.add_argument("--init", default=None, help="S1/S2 run folder (its best.pt) to start S3 from")
    ap.add_argument("--branches", default=None, help="folder with <prefix>EEG/, ... (best.pt + run.json)")
    ap.add_argument("--branch-prefix", default="", help="e.g. reg123_ for the regularised branches")
    # regularisation (xmai-lab/sleep-mechinterp#21, protocol reg123 of train_channelnet)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0, help="l2 (Adam weight_decay)")
    ap.add_argument("--sampler", default="stride", choices=["stride", "random"])
    ap.add_argument("--max-shift", type=int, default=300)
    ap.add_argument("--steps", type=int, default=None, help="step budget (overrides --passes)")
    ap.add_argument("--val-every", type=int, default=None, help="steps between validations (default 10 %% pass)")
    ap.add_argument("--select", default="loss", choices=["loss", "kappa"])
    ap.add_argument("--patience", type=int, default=10 ** 9, help="validations without improvement")
    ap.add_argument("--sched", default="const", choices=["const", "cosine"], help="S0-S2 (S3: warmup + cosine)")
    ap.add_argument("--warmup", type=int, default=None, help="S3 warmup steps (default one stride-1 pass)")
    ap.add_argument("--warmup-frac", type=float, default=None, help="S3 warmup as a fraction of the step budget")
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon (SleepTransformer paper: 1e-7)")
    ap.add_argument("--min-vals", type=int, default=0, help="validations before early stopping can trigger")
    ap.add_argument("--local-dir", default=None, help="node-local folder: nights read in parallel and memory-mapped")
    ap.add_argument("--workers", type=int, default=8, help="reader processes with --local-dir")
    ap.add_argument("--pipe", default="702cb90947921189", help="epoching pipeline where a channel has several")
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--occ-scale", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.3, help="weight of the per-branch auxiliary CE")
    ap.add_argument("--l2sp", type=float, default=0.0, help="S3: L2-SP weight towards pretrained branch weights")
    ap.add_argument("--hook", default="identity", choices=["identity", "detach", "grl"])
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--minutes", type=float, default=1e9)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-test-extras", action="store_true", help="only the full-modality test (smoke runs)")
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#19")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage == "S1" and args.points:
        ap.error("S1 is late fusion: no connection points")
    if (args.arch in ("ssn3ch", "st3ch")) != (args.stage == "S0"):
        ap.error("S0 is the stage of the 3-channel baselines --arch ssn3ch / st3ch (and only of them)")
    if args.arch == "mm" and not args.branches:
        ap.error("--branches required for --arch mm")

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
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, mods = args.seq_len, args.modalities

    if args.arch == "mm":
        branches, kappa = load_branches(Path(args.branches).expanduser(), mods, args.dropout, args.branch_prefix)
        model = MultiChannelNet(branches, args.points, hook=args.hook).to(device)
    else:
        kappa, args.aux, args.no_occlusion, args.skip_test_extras = None, 0.0, True, True
        model = (SeqSleepNet3chClean(mods, dropout=args.dropout) if args.arch == "ssn3ch"
                 else SleepTransformer3chClean(mods)).to(device)       # st3ch: native dropout 0.1
    if args.init:
        init = Path(args.init).expanduser() / "best.pt"
        model.load_state_dict(torch.load(init, map_location=device))
        log(f"initialised from {init}")
    theta0 = ({n: p.detach().clone() for n, p in model.branches.named_parameters()}
              if args.l2sp > 0 and args.arch == "mm" else None)

    lr_new, lr_branch = STAGE_LR[args.stage]
    if args.arch == "mm":
        model.frozen_branches = lr_branch is None       # S1/S2: frozen branches run without dropout
    for p in model.branch_parameters():
        p.requires_grad_(lr_branch is not None)
    groups = [{"params": model.new_parameters(), "lr": lr_new}]
    if lr_branch is not None:
        groups.append({"params": model.branch_parameters(), "lr": lr_branch})
    opt = torch.optim.Adam(groups, weight_decay=args.wd, eps=args.eps)
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    probs = None if args.no_occlusion else occlusion_probs(kappa, args.occ_scale)

    split = json.loads(Path(args.split_file).read_text()) if args.split_file else None
    data = CleanCacheMultiNights(args.root, args.datasets, mods, log=log, split=split, local_dir=args.local_dir,
                                 workers=args.workers, pipe=args.pipe)
    random_sampler = args.sampler == "random"
    if random_sampler:
        train_ds = MultiRandomOffsetSequences(data.nights["train"], L, max_shift=args.max_shift, seed=args.seed)
    else:
        train_ds = MultiSequenceDataset(data.nights["train"], L, stride=1)

    def loader():           # rebuilt every pass for the random sampler (the workers must see the new windows)
        return DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0 and not random_sampler, pin_memory=True)

    dl = loader()
    steps_total = args.steps or args.passes * len(dl)
    n_passes = math.ceil(steps_total / len(dl)) + 1 if args.steps else args.passes
    val_every = args.val_every or max(1, int(len(dl) * args.val_ratio))
    stride1_pass = sum(max(0, len(nt.y) - L + 1) for nt in data.nights["train"]) // args.batch
    if args.arch in ("ssn3ch", "st3ch"):   # input standardisation from ~200 training windows (Phan's scaling.npz)
        sdl = DataLoader(MultiSequenceDataset(data.nights["train"], L, stride=L), batch_size=16, shuffle=True)
        it = iter(sdl)
        model.fit_scaling([(to_dev(x, device), to_dev(o, device)) for x, _, o in (next(it) for _ in range(12))])
        log(f"input scaling fitted: mean dB {model.mean.mean(1).tolist()}, std {model.std.mean(1).tolist()}")

    def lr_factor(step):
        if args.stage != "S3":
            if args.sched == "cosine":
                return 0.5 * (1 + math.cos(math.pi * min(step, steps_total) / steps_total))
            return 1.0
        warm = int(args.warmup_frac * steps_total) if args.warmup_frac else (args.warmup or stride1_pass)
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps_total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = {"issue": args.issue, "git_sha": git_sha(), "config": cfg, "config_hash": cfg_hash,
           "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "device": str(device),
           "sites": model.sites, "n_trainable": n_train, "pretrained_kappa": kappa, "occlusion_probs": probs,
           "nights": {k: len(v) for k, v in data.nights.items()}, "skipped": data.skipped,
           "train_class_counts": multi_class_counts(data.nights["train"]).tolist(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log(f"stage {args.stage}, sites {model.sites}, {n_train} trainable params, {len(train_ds)} windows, "
        f"{len(dl)} steps/pass, validation every {val_every}, occlusion {probs}")

    v0 = tiled_eval(model, data.nights["valid"], device, L, crit)
    ref_alone = v0["alone_kappa"]
    log(f"init valid: loss {v0['loss']:.4f} κ {v0['kappa']:.4f} alone {ref_alone}")
    mf = open(out / "metrics.jsonl", "a")
    mf.write(json.dumps({"step": 0, "valid": v0, "gates": model.gates()}) + "\n")

    def score_of(v):         # higher is better
        return v["kappa"] if args.select == "kappa" else -v["loss"]

    t0, step, best, stop, run_loss, bad, curve, n_vals = time.time(), 0, None, False, [], 0, [], 0
    if args.init:            # S3: the starting checkpoint competes for best.pt, so fine-tuning can never end worse
        best = score_of(v0)
        torch.save(model.state_dict(), out / "best.pt")
    for ep in range(n_passes):
        if random_sampler and ep > 0:
            train_ds.set_epoch(ep)
            dl = loader()
        model.train()
        for x, y, off in dl:
            x, off, y = to_dev(x, device), to_dev(off, device), y.to(device)
            occ = None if probs is None else sample_occlusion(y.shape[0], probs, mods, device)
            logits, aux, _ = model(x, off, occ)
            loss = crit(logits.reshape(-1, 5), y.reshape(-1))
            if args.aux > 0:
                loss = loss + args.aux * sum(crit(aux[m].reshape(-1, 5), y.reshape(-1)) for m in mods)
            if theta0 is not None:
                loss = loss + args.l2sp * sum(((p - theta0[n]) ** 2).sum()
                                              for n, p in model.branches.named_parameters())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss.append(float(loss))
            timeout = time.time() - t0 > args.minutes * 60
            if step % val_every == 0 or timeout:
                v = tiled_eval(model, data.nights["valid"], device, L, crit)
                forget = {m: v["alone_kappa"][m] - ref_alone[m] for m in v["alone_kappa"]}
                curve.append((step, ep, float(np.mean(run_loss[-val_every:])), v["loss"], v["kappa"]))
                rec = {"step": step, "pass": ep, "min": round((time.time() - t0) / 60, 2),
                       "train_loss": float(np.mean(run_loss[-val_every:])), "valid": v, "forgetting": forget,
                       "gates": model.gates(), "lr": [g["lr"] for g in opt.param_groups]}
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                gmax = max([abs(g) for s in rec["gates"].values() for g in s.values()] or [0.0])
                log(f"step {step} pass {ep} loss {rec['train_loss']:.4f} val_loss {v['loss']:.4f} κ {v['kappa']:.4f} "
                    f"forget {{{', '.join(f'{m}: {d:+.3f}' for m, d in forget.items())}}} |g|max {gmax:.3f}")
                if best is None or score_of(v) > best:
                    best, bad = score_of(v), 0
                    torch.save(model.state_dict(), out / "best.pt")
                else:
                    bad += 1
                model.train()
                n_vals += 1
                if timeout or (n_vals >= args.min_vals and bad >= args.patience):
                    stop = True
                    break
            if step >= steps_total:
                stop = True
                break
        if stop:
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    test = data.nights["test"]
    final = {"steps": step, "minutes": round((time.time() - t0) / 60, 2),
             "valid_best": {"select": args.select, "value": best if args.select == "kappa" else -best},
             "gates": model.gates(), "test": voting(model, test, device, L)}
    if curve:               # overfitting diagnostics over the validation curve (as train_channelnet)
        c = np.array(curve)
        ib, il = int(c[:, 4].argmax()), int(c[:, 3].argmin())
        final["overfit"] = {"best_kappa_step": int(c[ib, 0]), "max_val_kappa": float(c[ib, 4]),
                            "final_val_kappa": float(c[-1, 4]), "kappa_drop": float(c[ib, 4] - c[-1, 4]),
                            "min_val_loss_step": int(c[il, 0]), "min_val_loss": float(c[il, 3]),
                            "final_val_loss": float(c[-1, 3]), "final_train_loss": float(c[-1, 2])}
    log(f"test κ {final['test']['kappa']:.4f}")
    if not args.skip_test_extras:
        final["test_subsets"] = {}
        for r in range(1, len(mods)):
            for keep in itertools.combinations(mods, r):
                occ = set(mods) - set(keep)
                k = "+".join(keep)
                final["test_subsets"][k] = {kk: vv for kk, vv in voting(model, test, device, L, occlude=occ).items()
                                            if kk in ("kappa", "acc", "macro_f1")}
                log(f"subset {k}: κ {final['test_subsets'][k]['kappa']:.4f}")
        final["test_ablation"] = {}
        for pt in ("F", "E", "Q"):
            sites = [s for s in model.sites if s.startswith(pt)]
            if sites:
                kk = voting(model, test, device, L, disabled=sites)["kappa"]
                final["test_ablation"][pt] = {"kappa": kk, "delta": kk - final["test"]["kappa"]}
                log(f"ablation {pt} off: κ {kk:.4f}")
        final["test_alone"] = {m: voting(model, test, device, L, disabled=model.sites, head=m)["kappa"] for m in mods}
        final["test_forgetting"] = {m: final["test_alone"][m] - kappa[m] for m in mods}
        log(f"alone {final['test_alone']} forgetting {final['test_forgetting']}")
    run["final"] = final
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
