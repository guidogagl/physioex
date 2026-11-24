"""Train one single-channel ChannelNet (corrected SeqSleepNet) on the clean cache.

    python -m physioex.train.bin.train_channelnet --modality EEG --root ~/work/cache/samples/v1 \
        --datasets mass_ss01 mass_ss03 --minutes 5 --out <run folder>

Protocols:
- ``proof`` (default): AdamW, class-weighted CE, early stopping on validation κ, time cap (quick checks).
- ``phan``: the protocol of ``examples/pretrained/seqsleepnet-phan`` (physioex Trainer): L = 20, Adam lr 1e-4,
  weight decay 0, unweighted CE, batch 32, 10 passes over training windows taken at every epoch (stride 1),
  validation every 10 % of a pass, best checkpoint by validation loss, test with ``Trainer.voting_evaluate``
  (L offsets, averaged logits).
- ``reg1`` / ``reg12`` / ``reg123``: phan plus, cumulatively, (1) dropout 0.1 + l2 1e-4; (2) windows re-cut at a
  random offset every pass with a +-300-sample time shift (``RandomOffsetSequences``), budget 48,760 steps (= phan),
  validation every 487 steps; (3) cosine learning-rate decay, best checkpoint by validation kappa, early stopping
  after 20 validations (xmai-lab/sleep-mechinterp#21).

--split-file takes {dataset: {train, valid, test}} subject lists (e.g. physioex fold 0).
--overfit N trains on the first N training nights only and reports training loss (pipeline check).
Writes run.json (git sha, config hash, host, issue), metrics.jsonl (one line per validation) and best.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import math

from physioex.data.clean_cache import CleanCacheNights, RandomOffsetSequences, SequenceDataset, class_counts
from physioex.models.seqsleepnet_clean import ChannelNet, saturation

_REG = dict(seq_len=20, stride=1, batch=32, lr=1e-4, wd=1e-4, optim="adam", weighted=False, dropout=0.1)
PROTOCOLS = {
    "proof": dict(seq_len=21, stride=10, batch=32, lr=1e-3, wd=1e-4, optim="adamw", weighted=True,
                  max_passes=100, select="kappa"),
    "phan": dict(seq_len=20, stride=1, batch=32, lr=1e-4, wd=0.0, optim="adam", weighted=False,
                 max_passes=10, select="loss", val_ratio=0.1, patience=10, patience_unit="pass"),
    "reg1": dict(_REG, max_passes=10, select="loss", val_ratio=0.1, patience=10, patience_unit="pass"),
    "reg12": dict(_REG, sampler="random", max_shift=300, steps=48760, val_every=487, select="loss",
                  patience=10 ** 9, patience_unit="val"),
    "reg123": dict(_REG, sampler="random", max_shift=300, steps=48760, val_every=487, select="kappa",
                   sched="cosine", patience=20, patience_unit="val"),
}


def git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _metrics(yt, yp):
    return {"acc": float(accuracy_score(yt, yp)), "kappa": float(cohen_kappa_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=range(5))),
            "f1_per_class": [float(v) for v in f1_score(yt, yp, average=None, labels=range(5))],
            "confusion": confusion_matrix(yt, yp, labels=range(5)).tolist(), "n_epochs": int(len(yt))}


@torch.no_grad()
def evaluate(model, nights, device, L, crit=None, bs=64):
    """Tiled windows (stride L, last aligned to the end): metrics and mean loss. Used for validation."""
    model.eval()
    ds = SequenceDataset(nights, L, stride=L)
    pred = [np.full(len(n.y), -1) for n in nights]
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    k, losses, conf, hit = 0, [], [], []
    for x, y, off in dl:
        logits, _ = model(x.to(device), off.to(device))
        if crit is not None:
            losses.append(float(crit(logits.reshape(-1, 5), y.to(device).reshape(-1))))
        pr, yy = logits.softmax(-1).reshape(-1, 5).cpu(), y.reshape(-1)
        msk = yy >= 0
        conf.append(pr.max(-1).values[msk].numpy())
        hit.append((pr.argmax(-1) == yy)[msk].numpy())
        for row in logits.argmax(-1).cpu().numpy():
            ni, s = ds.windows[k]
            pred[ni][s:s + L] = row
            k += 1
    yt, yp = np.concatenate([n.y for n in nights]), np.concatenate(pred)
    m = (yt >= 0) & (yp >= 0)
    res = _metrics(yt[m], yp[m])
    if losses:
        res["loss"] = float(np.mean(losses))
    res["ece"] = ece(np.concatenate(conf), np.concatenate(hit))
    return res


def ece(conf, hit, bins=15):
    """Expected calibration error (equal-width confidence bins) over scored validation epochs (tiled windows)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - hit[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


@torch.no_grad()
def voting_evaluate(model, nights, device, L, bs=64):
    """As physioex Trainer.voting_evaluate: for each offset 0..L-1 the night is cut into consecutive windows of L,
    the logits of all windows covering an epoch are averaged, argmax per epoch."""
    model.eval()
    yt_all, yp_all, per_night = [], [], []
    for nt in nights:
        n = len(nt.y)
        if n < L:
            continue
        x = torch.from_numpy(np.asarray(nt.x, dtype=np.float32))
        off = torch.tensor([nt.offset_db], dtype=torch.float32, device=device)
        votes = torch.zeros(n, 5, device=device)
        counts = torch.zeros(n, device=device)
        for o in range(L):
            usable = (n - o) - (n - o) % L
            if usable == 0:
                continue
            xw = x[o:o + usable].reshape(-1, L, x.shape[-1])
            for i in range(0, len(xw), bs):
                xb = xw[i:i + bs].to(device)
                logits, _ = model(xb, off.expand(len(xb)))
                a = o + i * L
                votes[a:a + logits.shape[0] * L] += logits.reshape(-1, 5).float()
                counts[a:a + logits.shape[0] * L] += 1
        yp = (votes / counts.clamp(min=1).unsqueeze(-1)).argmax(-1).cpu().numpy()
        m = nt.y >= 0
        yt_all.append(nt.y[m])
        yp_all.append(yp[m])
        per_night.append({"dataset": nt.dataset, "kappa": float(cohen_kappa_score(nt.y[m], yp[m])),
                          "acc": float(accuracy_score(nt.y[m], yp[m])), "n": int(m.sum())})
    res = _metrics(np.concatenate(yt_all), np.concatenate(yp_all))
    res["per_dataset"] = {}
    for ds in sorted({p["dataset"] for p in per_night}):
        sel = [i for i, p in enumerate(per_night) if p["dataset"] == ds]
        yt = np.concatenate([yt_all[i] for i in sel])
        yp = np.concatenate([yp_all[i] for i in sel])
        res["per_dataset"][ds] = {"kappa": float(cohen_kappa_score(yt, yp)), "acc": float(accuracy_score(yt, yp)),
                                  "nights": len(sel)}
    res["night_kappa_mean"] = float(np.mean([p["kappa"] for p in per_night]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["EEG", "EOG", "EMG"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="proof")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--seq-len", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--wd", type=float)
    ap.add_argument("--max-passes", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--sampler", choices=["stride", "random"])
    ap.add_argument("--max-shift", type=int)
    ap.add_argument("--sched", choices=["const", "cosine"])
    ap.add_argument("--select", choices=["loss", "kappa"])
    ap.add_argument("--steps", type=int, help="step budget (overrides --max-passes)")
    ap.add_argument("--minutes", type=float, default=1e9)
    ap.add_argument("--val-every", type=int, default=200, help="steps between validations (proof protocol)")
    ap.add_argument("--patience", type=int, default=5, help="validations without improvement")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--deformable", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#12")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    P = dict(PROTOCOLS[args.protocol])
    for k in ("seq_len", "stride", "batch", "lr", "wd", "max_passes", "dropout", "sampler", "max_shift", "sched",
              "select", "steps"):
        if getattr(args, k) is not None:
            P[k] = getattr(args, k)
    if "patience" in P and args.patience == 5:
        args.patience = P["patience"]

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
    cfg["resolved"] = P
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = P["seq_len"]

    split = json.loads(Path(args.split_file).read_text()) if args.split_file else None
    data = CleanCacheNights(args.root, args.datasets, args.modality, log=log, split=split)
    train_n = data.nights["train"][: args.overfit] if args.overfit else data.nights["train"]
    counts = class_counts(train_n)
    w = None
    if P["weighted"]:
        w = torch.tensor(counts.sum() / np.maximum(counts, 1) / 5, dtype=torch.float32)
        w = (w / w.mean()).to(device)
    log(f"train class counts {counts.tolist()}, CE weights {None if w is None else [round(float(v), 2) for v in w]}")

    model = ChannelNet(args.modality, deformable=args.deformable, dropout=P.get("dropout", 0.0)).to(device)
    if P["optim"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=P["lr"], weight_decay=P["wd"])
    else:
        lstm1 = [p for n, p in model.named_parameters() if n.startswith("epoch_rnn.rnns.0.weight_ih")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("epoch_rnn.rnns.0.weight_ih")]
        opt = torch.optim.AdamW([{"params": lstm1, "weight_decay": 10 * P["wd"]},
                                 {"params": rest, "weight_decay": P["wd"]}], lr=P["lr"])
    crit = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    n_par = sum(p.numel() for p in model.parameters())

    run = {"issue": args.issue, "git_sha": git_sha(), "config": cfg, "config_hash": cfg_hash,
           "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "device": str(device),
           "n_params": n_par, "nights": {k: len(v) for k, v in data.nights.items()}, "skipped": data.skipped,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run.json").write_text(json.dumps(run, indent=2))

    random_sampler = P.get("sampler") == "random"
    if random_sampler:
        train_ds = RandomOffsetSequences(train_n, L, max_shift=P.get("max_shift", 300), seed=args.seed)
    else:
        train_ds = SequenceDataset(train_n, L, stride=P["stride"])

    def loader():           # rebuilt every pass for the random sampler (the workers must see the new windows)
        return DataLoader(train_ds, batch_size=P["batch"], shuffle=True, num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0 and not random_sampler)

    dl = loader()
    val_every = (max(1, int(len(dl) * P["val_ratio"])) if "val_ratio" in P
                 else P.get("val_every", args.val_every))
    total_steps = P.get("steps") or P["max_passes"] * len(dl)
    max_passes = math.ceil(total_steps / len(dl)) + 1 if P.get("steps") else P["max_passes"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda st: 0.5 * (1 + math.cos(math.pi * min(st, total_steps) / total_steps)))
        if P.get("sched") == "cosine" else (lambda st: 1.0))
    per_pass = P.get("patience_unit") == "pass"
    curve = []
    log(f"{args.modality}: {n_par} params, {len(train_ds)} train windows, {len(dl)} steps/pass, "
        f"validation every {val_every} steps, device {device}")

    better = (lambda a, b: a < b) if P["select"] == "loss" else (lambda a, b: a > b)
    t0, step, best, bad, stop = time.time(), 0, None, 0, False
    improved, bad_passes = False, 0
    probe = next(iter(DataLoader(SequenceDataset(data.nights["valid"] or train_n, L), batch_size=16)))
    mf = open(out / "metrics.jsonl", "a")
    sat0 = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
    log(f"saturation at init {sat0}")
    run_loss = []
    for ep in range(max_passes):
        if random_sampler and ep > 0:
            train_ds.set_epoch(ep)
            dl = loader()
        model.train()
        for x, y, off in dl:
            logits, _ = model(x.to(device), off.to(device))
            loss = crit(logits.reshape(-1, 5), y.to(device).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.protocol == "proof":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            run_loss.append(float(loss))
            timeout = time.time() - t0 > args.minutes * 60
            if step % val_every == 0 or timeout:
                rec = {"step": step, "pass": ep, "min": round((time.time() - t0) / 60, 2),
                       "train_loss": float(np.mean(run_loss[-val_every:]))}
                if not args.overfit:
                    rec["valid"] = evaluate(model, data.nights["valid"], device, L, crit)
                rec["sat"] = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
                model.train()
                rec["lr"] = opt.param_groups[0]["lr"]
                if "valid" in rec:
                    curve.append((step, ep, rec["train_loss"], rec["valid"]["loss"], rec["valid"]["kappa"]))
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                v = rec.get("valid", {})
                score = rec["train_loss"] if args.overfit else (v["loss"] if P["select"] == "loss" else v["kappa"])
                if args.overfit and P["select"] != "loss":
                    score = -score
                log(f"step {step} pass {ep} loss {rec['train_loss']:.4f} val_loss {v.get('loss', float('nan')):.4f} "
                    f"κ {v.get('kappa', float('nan')):.4f} sat L1 {rec['sat']['h_sat_L1']:.3f}/"
                    f"{rec['sat']['gate_sat_L1']:.3f}")
                if best is None or better(score, best):
                    best, bad, improved = score, 0, True
                    torch.save(model.state_dict(), out / "best.pt")
                else:
                    bad += 1
                # proof: patience in validations; phan: in passes, as the physioex Trainer (checked below)
                if timeout or (not per_pass and bad >= args.patience) or step >= total_steps:
                    stop = True
                    break
            if step >= total_steps:
                stop = True
                break
        if per_pass:
            bad_passes = 0 if improved else bad_passes + 1
            improved = False
            if bad_passes >= args.patience:
                stop = True
        if stop:
            break

    final = {"steps": step, "passes": ep + 1, "minutes": round((time.time() - t0) / 60, 2), "sat_init": sat0,
             "train_loss_last": float(np.mean(run_loss[-50:]))}
    if curve:               # overfitting diagnostics over the validation curve
        c = np.array(curve)
        ib, il = int(c[:, 4].argmax()), int(c[:, 3].argmin())
        final["overfit"] = {"best_kappa_step": int(c[ib, 0]), "best_kappa_pass": int(c[ib, 1]),
                            "max_val_kappa": float(c[ib, 4]), "final_val_kappa": float(c[-1, 4]),
                            "kappa_drop": float(c[ib, 4] - c[-1, 4]),
                            "min_val_loss_step": int(c[il, 0]), "min_val_loss": float(c[il, 3]),
                            "final_val_loss": float(c[-1, 3]), "final_train_loss": float(c[-1, 2]),
                            "final_gap": float(c[-1, 3] - c[-1, 2])}
    if not args.overfit:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        final["valid_best"] = {"select": P["select"], "value": best}
        final["test_voting"] = voting_evaluate(model, data.nights["test"], device, L)
        final["test"] = final["test_voting"]
        final["sat_best"] = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
    run["final"] = final
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log(f"final {json.dumps({k: v for k, v in final.items() if k != 'test'})}")
    log("DONE")


if __name__ == "__main__":
    main()
