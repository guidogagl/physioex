"""SHHS-1 split of the SleepTransformer framework runs (xmai-lab/sleep-mechinterp#26).

Starts from the physioex SHHS benchmark split (``physioex/data/datasets/_splits/shhs.json`` fold 0, the split of
Phan et al.: 4065 train / 100 valid / 1639 test), so the ``sleeptransformer-phan`` reference shares the test subjects.
Applies the paper filter: a night is kept only if its labels (clean cache v1) contain all five stages. Nights missing
from the cache are dropped. Writes ``{"shhs_visit1": {train, valid, test}}`` and a ``.counts.json`` beside it.

    python examples/pretrained/st-shhs/make_split.py --root <cache v1> --out examples/pretrained/st-shhs/shhs1_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[3] / "physioex/data/datasets/_splits/shhs.json"


def has_five_stages(y: np.ndarray) -> bool:
    return set(range(5)) <= set(np.unique(y[y >= 0]).tolist())


def make_split(root: Path, ds: str = "shhs_visit1", bench: Path = BENCH):
    fold = json.loads(bench.read_text())["fold_0"]
    split, counts = {}, {}
    for k, ids in fold.items():
        keep, missing, partial = [], 0, 0
        for s in ids:
            f = root / ds / "labels" / f"{s}.npy"
            if not f.exists() or not (root / ds / "signals" / s).is_dir():
                missing += 1
            elif not has_five_stages(np.load(f)):
                partial += 1
            else:
                keep.append(s)
        split[k] = keep
        counts[k] = {"benchmark": len(ids), "kept": len(keep), "not_in_cache": missing, "missing_stage": partial}
    return {ds: split}, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split, counts = make_split(Path(args.root))
    out = Path(args.out)
    out.write_text(json.dumps(split, indent=1))
    n = sum(c["kept"] for c in counts.values())
    counts["ratio"] = {k: round(c["kept"] / n, 4) for k, c in counts.items()}
    out.with_suffix(".counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
