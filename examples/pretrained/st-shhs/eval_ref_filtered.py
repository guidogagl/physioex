"""sleeptransformer-phan (raw C4-A1 EEG, physioex seqsleepnet pipeline) on the five-stage-filtered SHHS-1 test split
of the framework runs (shhs1_split.json, 1532 nights), voting L 21: the reference re-scored on the same test nights
as ST-S3F and st3ch (xmai-lab/sleep-mechinterp#28).

    python examples/pretrained/st-shhs/eval_ref_filtered.py --out <folder>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--split", default=str(Path(__file__).with_name("shhs1_split.json")))
args = ap.parse_args()
ids = set(json.loads(Path(args.split).read_text())["shhs_visit1"]["test"])
model = load_from_pretrained("sleeptransformer-phan")
D = get_dataset("shhs")(channels=["EEG"], pipelines="seqsleepnet", sequence_length=21)
orig = D.split


def split(fold=0):
    tr, va, te = orig(fold=fold)
    keep = [t for t in te if str(t[-1]) in ids or f"shhs1-{t[-1]}" in ids]
    print(f"test subjects {len(te)} -> {len(keep)} kept (example id {te[0][-1]!r})", flush=True)
    if abs(len(keep) - len(ids)) > 10:
        raise SystemExit(f"id matching failed: {len(keep)} of {len(ids)}")
    return tr, va, keep


D.split = split
res = Trainer.voting_evaluate(model=model, dataset=D, L=21, fold=0, gpu_id=0 if torch.cuda.is_available() else None)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
clean = {k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v, "item") else v)) for k, v in res.items()}
(out / "metrics_filtered.json").write_text(json.dumps(clean, indent=1, default=str))
print({k: v for k, v in clean.items() if isinstance(v, float)}, flush=True)
