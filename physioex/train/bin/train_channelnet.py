"""Train one single-channel ChannelNet (corrected SeqSleepNet) on the clean cache.

    python -m physioex.train.bin.train_channelnet --modality EEG --root ~/work/cache/samples/v1 \
        --datasets mass_ss01 mass_ss03 --minutes 5 --out <run folder>

--overfit N trains on the first N training nights only and reports training loss (pipeline check).
Writes run.json (git sha, config hash, host, issue), metrics.jsonl (one line per validation) and best.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

from physioex.data.clean_cache import CleanCacheNights, SequenceDataset, class_counts
from physioex.models.seqsleepnet_clean import ChannelNet, saturation


def git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate(model, nights, device, L, bs=64):
    model.eval()
    ds = SequenceDataset(nights, L, stride=L)
    pred = [np.full(len(n.y), -1) for n in nights]
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    k = 0
    for x, y, off in dl:
        logits, _ = model(x.to(device), off.to(device))
        p = logits.argmax(-1).cpu().numpy()
        for row in p:
            ni, s = ds.windows[k]
            pred[ni][s:s + L] = row
            k += 1
    yt = np.concatenate([n.y for n in nights])
    yp = np.concatenate(pred)
    m = (yt >= 0) & (yp >= 0)
    yt, yp = yt[m], yp[m]
    return {"acc": float(accuracy_score(yt, yp)), "kappa": float(cohen_kappa_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=range(5))),
            "f1_per_class": [float(v) for v in f1_score(yt, yp, average=None, labels=range(5))],
            "n_epochs": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["EEG", "EOG", "EMG"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=21)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--max-passes", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=200, help="steps between validations")
    ap.add_argument("--patience", type=int, default=5, help="validations without κ gain")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--deformable", action="store_true")
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#12")
    ap.add_argument("--seed", type=int, default=42)
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
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CleanCacheNights(args.root, args.datasets, args.modality, log=log)
    train_n = data.nights["train"][: args.overfit] if args.overfit else data.nights["train"]
    counts = class_counts(train_n)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / 5, dtype=torch.float32)
    w = (w / w.mean()).to(device)
    log(f"train class counts {counts.tolist()}, CE weights {[round(float(v), 2) for v in w]}")

    model = ChannelNet(args.modality, deformable=args.deformable).to(device)
    lstm1 = [p for n, p in model.named_parameters() if n.startswith("epoch_rnn.rnns.0.weight_ih")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("epoch_rnn.rnns.0.weight_ih")]
    opt = torch.optim.AdamW([{"params": lstm1, "weight_decay": 10 * args.wd},
                             {"params": rest, "weight_decay": args.wd}], lr=args.lr)
    crit = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    n_par = sum(p.numel() for p in model.parameters())

    run = {"issue": args.issue, "git_sha": git_sha(), "config": cfg, "config_hash": cfg_hash,
           "host": socket.gethostname(), "device": str(device), "n_params": n_par,
           "nights": {k: len(v) for k, v in data.nights.items()}, "skipped": data.skipped,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run.json").write_text(json.dumps(run, indent=2))

    train_ds = SequenceDataset(train_n, args.seq_len, stride=args.stride)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    log(f"{args.modality}: {n_par} params, {len(train_ds)} train windows, device {device}")

    t0, step, best, bad, stop = time.time(), 0, -1.0, 0, False
    probe = next(iter(DataLoader(SequenceDataset(data.nights["valid"] or train_n, args.seq_len), batch_size=16)))
    mf = open(out / "metrics.jsonl", "a")
    sat0 = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
    log(f"saturation at init {sat0}")
    run_loss = []
    for ep in range(args.max_passes):
        model.train()
        for x, y, off in dl:
            logits, _ = model(x.to(device), off.to(device))
            loss = crit(logits.reshape(-1, 5), y.to(device).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            run_loss.append(float(loss))
            timeout = time.time() - t0 > args.minutes * 60
            if step % args.val_every == 0 or timeout:
                rec = {"step": step, "pass": ep, "min": round((time.time() - t0) / 60, 2),
                       "train_loss": float(np.mean(run_loss[-args.val_every:]))}
                if not args.overfit:
                    rec["valid"] = evaluate(model, data.nights["valid"], device, args.seq_len)
                rec["sat"] = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
                model.train()
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                kv = rec.get("valid", {}).get("kappa", -rec["train_loss"])
                log(f"step {step} pass {ep} loss {rec['train_loss']:.4f} "
                    f"κ {rec.get('valid', {}).get('kappa', float('nan')):.4f} sat {rec['sat']}")
                if kv > best:
                    best, bad = kv, 0
                    torch.save(model.state_dict(), out / "best.pt")
                else:
                    bad += 1
                if timeout or bad >= args.patience:
                    stop = True
                    break
        if stop:
            break

    final = {"steps": step, "minutes": round((time.time() - t0) / 60, 2), "sat_init": sat0,
             "train_loss_last": float(np.mean(run_loss[-50:]))}
    if not args.overfit:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        final["valid_best_kappa"] = best
        final["test"] = evaluate(model, data.nights["test"], device, args.seq_len)
        final["sat_best"] = saturation(model.eval(), probe[0].to(device), probe[2].to(device))
    run["final"] = final
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log(f"final {json.dumps(final)}")
    log("DONE")


if __name__ == "__main__":
    main()
