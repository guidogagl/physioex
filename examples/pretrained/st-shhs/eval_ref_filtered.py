"""sleeptransformer-phan (raw C4-A1 EEG, physioex seqsleepnet pipeline) on the five-stage-filtered SHHS-1 test split
of the framework runs (shhs1_split.json, 1532 nights), voting L 21: the reference re-scored on the same test nights
as ST-S3F and st3ch (xmai-lab/sleep-mechinterp#28).

One voting_evaluate call per test subject, so every prediction is tied to its subject id (the order of the shared
eval loader is not guaranteed). Each process scores subjects [--start, --stop) and writes part_<start>.npz (the
dataset leaks open files across calls and GPFS stalls after ~700 calls, so the sbatch runs chunks of 250 in fresh
processes); --merge combines the parts. Writes metrics_filtered.json (pooled metrics, per-night kappa) and preds.npz
({id}|y, {id}|p, ids) for paired comparisons with the clean-cache models.

    python examples/pretrained/st-shhs/eval_ref_filtered.py --out <folder>
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train.bin.train_channelnet import _metrics
from physioex.train.trainer import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--split", default=str(Path(__file__).with_name("shhs1_split.json")))
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=None)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.merge:
    store = {}
    for f in sorted(out.glob("part_*.npz")):
        z = np.load(f)
        store.update({k: z[k] for k in z.files if k != "ids"})
    done = sorted({k.split("|")[0] for k in store})
    yt = np.concatenate([store[f"{s}|y"] for s in done])
    yp = np.concatenate([store[f"{s}|p"] for s in done])
    m = _metrics(yt, yp)
    m["per_night_kappa"] = {s: float(cohen_kappa_score(store[f"{s}|y"], store[f"{s}|p"])) for s in done}
    m["nights"] = len(done)
    np.savez_compressed(out / "preds.npz", ids=np.array(done), **store)
    (out / "metrics_filtered.json").write_text(json.dumps(m, indent=1, default=float))
    print({k: v for k, v in m.items() if isinstance(v, (int, float))}, flush=True)
    print("DONE", flush=True)
    raise SystemExit(0)
ids = set(json.loads(Path(args.split).read_text())["shhs_visit1"]["test"])
model = load_from_pretrained("sleeptransformer-phan")
D = get_dataset("shhs")(channels=["EEG"], pipelines="seqsleepnet", sequence_length=21)
orig = D.split
tr, va, te = orig(fold=0)


def sid(t):
    s = str(t[-1])
    return s if s.startswith("shhs1-") else f"shhs1-{s}"


keep = sorted((t for t in te if sid(t) in ids), key=sid)
print(f"test subjects {len(te)} -> {len(keep)} kept (example id {te[0][-1]!r})", flush=True)
if abs(len(keep) - len(ids)) > 10:
    raise SystemExit(f"id matching failed: {len(keep)} of {len(ids)}")

cap = {}


def capture(preds, targets, ignore_index=-1):
    cap["p"], cap["y"] = preds.argmax(-1).numpy().reshape(-1), targets.numpy().reshape(-1)
    return 0.0


gpu = 0 if torch.cuda.is_available() else None
store, done, t0 = {}, [], time.time()
for i, t in enumerate(keep[args.start:args.stop], start=args.start):
    D.split = lambda fold=0, t=t: (tr[:1], va[:1], [t])    # tiny train/valid lists: only the test loader is used
    cap.clear()
    Trainer.voting_evaluate(model=model, dataset=D, L=21, fold=0, gpu_id=gpu, metrics={"cap": capture})
    D._memmap_cache.clear()
    gc.collect()
    if "y" not in cap:                                       # skipped by voting_evaluate (shorter than L)
        continue
    ok = cap["y"] >= 0
    store[f"{sid(t)}|y"], store[f"{sid(t)}|p"] = cap["y"][ok], cap["p"][ok]
    done.append(sid(t))
    if i % 100 == 0:
        print(f"{i}/{len(keep)} {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(out / f"part_{args.start:05d}.npz", ids=np.array(done), **store)
print(f"part {args.start}: {len(done)} nights, {time.time() - t0:.0f}s", flush=True)
