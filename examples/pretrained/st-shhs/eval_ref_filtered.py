"""sleeptransformer-phan (raw C4-A1 EEG, physioex seqsleepnet pipeline) on the five-stage-filtered SHHS-1 test split
of the framework runs (shhs1_split.json, 1532 nights), voting L 21: the reference re-scored on the same test nights
as ST-S3F and st3ch (xmai-lab/sleep-mechinterp#28).

One voting_evaluate call per test subject, so every prediction is tied to its subject id (the order of the shared
eval loader is not guaranteed). Writes metrics_filtered.json (pooled metrics, per-night kappa) and preds.npz
({id}|y, {id}|p, ids) for paired comparisons with the clean-cache models.

    python examples/pretrained/st-shhs/eval_ref_filtered.py --out <folder>
"""

import argparse
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
args = ap.parse_args()
ids = set(json.loads(Path(args.split).read_text())["shhs_visit1"]["test"])
model = load_from_pretrained("sleeptransformer-phan")
D = get_dataset("shhs")(channels=["EEG"], pipelines="seqsleepnet", sequence_length=21)
orig = D.split
tr, va, te = orig(fold=0)


def sid(t):
    s = str(t[-1])
    return s if s.startswith("shhs1-") else f"shhs1-{s}"


keep = [t for t in te if sid(t) in ids]
print(f"test subjects {len(te)} -> {len(keep)} kept (example id {te[0][-1]!r})", flush=True)
if abs(len(keep) - len(ids)) > 10:
    raise SystemExit(f"id matching failed: {len(keep)} of {len(ids)}")

cap = {}


def capture(preds, targets, ignore_index=-1):
    cap["p"], cap["y"] = preds.argmax(-1).numpy().reshape(-1), targets.numpy().reshape(-1)
    return 0.0


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
gpu = 0 if torch.cuda.is_available() else None
store, done, t0 = {}, [], time.time()
for i, t in enumerate(keep):
    D.split = lambda fold=0, t=t: (tr[:1], va[:1], [t])    # tiny train/valid lists: only the test loader is used
    cap.clear()
    Trainer.voting_evaluate(model=model, dataset=D, L=21, fold=0, gpu_id=gpu, metrics={"cap": capture})
    if "y" not in cap:                                       # skipped by voting_evaluate (shorter than L)
        continue
    ok = cap["y"] >= 0
    store[f"{sid(t)}|y"], store[f"{sid(t)}|p"] = cap["y"][ok], cap["p"][ok]
    done.append(sid(t))
    if i % 100 == 0:
        print(f"{i}/{len(keep)} {time.time() - t0:.0f}s", flush=True)

yt = np.concatenate([store[f"{s}|y"] for s in done])
yp = np.concatenate([store[f"{s}|p"] for s in done])
m = _metrics(yt, yp)
m["per_night_kappa"] = {s: float(cohen_kappa_score(store[f"{s}|y"], store[f"{s}|p"])) for s in done}
m.update({"nights": len(done), "minutes": round((time.time() - t0) / 60, 1)})
np.savez_compressed(out / "preds.npz", ids=np.array(done), **store)
(out / "metrics_filtered.json").write_text(json.dumps(m, indent=1, default=float))
print({k: v for k, v in m.items() if isinstance(v, float)}, flush=True)
print("DONE", flush=True)
