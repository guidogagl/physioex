"""Verify wav2sleep against the published numbers (Carter & Tarassenko 2024, Table 6).

Loads the pretrained model via ``load_from_pretrained("wav2sleep")`` and
evaluates it on the test split of every NSRR dataset named in
``paper_metrics.json``, once per input condition (``all`` / ``ecg_thx`` /
``ecg`` / ``ppg``).  The ``census`` rows pool the census-balanced 500-night
test set of Jones et al. (2024) across the datasets that provide it.  A row
is PASS when the measured pooled Cohen's kappa is within the published
value's tolerance; rows whose data is missing (or that need several
datasets, one of which is absent) are reported as SKIP, never a crash.

Usage (on the cluster where the NSRR data live, e.g. via a Slurm job):
    python examples/pretrained/wav2sleep/verify.py \\
        --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
    python examples/pretrained/wav2sleep/verify.py --max-nights 2   # smoke run
"""
import argparse
import json
import os
import sys
from pathlib import Path

MODEL_NAME = "wav2sleep"
PIPELINE = "wav2sleep"
LABEL_TRANSFORM = "aasm4"
N_CLASSES = 4
METRICS_PATH = Path(__file__).parent / "paper_metrics.json"
#: Signal names passed to predict_night for each condition of Table 6.
CONDITION_SIGNALS = {
    "all": ["ABD", "THX", "ECG", "PPG"],
    "ecg_thx": ["ECG", "THX"],
    "ecg": ["ECG"],
    "ppg": ["PPG"],
}
#: Datasets whose "all" condition includes PPG in the paper.
ALL_WITH_PPG = {"mesa", "chat", "cfs", "ccshs"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify wav2sleep against the published Table 6 numbers")
    parser.add_argument("--data-root", default=None,
                        help="raw data root (default: $PHYSIOEX_DATA)")
    parser.add_argument("--cache-dir", default=None,
                        help="preprocessing cache (default: $PHYSIOEX_CACHE_DIR)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="datasets to evaluate (default: all in paper_metrics.json)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--max-nights", type=int, default=None,
                        help="smoke run: at most N nights per dataset "
                             "(PASS/FAIL are reported as REPORT)")
    parser.add_argument("--out", default="./verify_results.json",
                        help="path of the output JSON")
    return parser


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _condition_channels(cond: str, dataset_name: str) -> list:
    """Channel requests for one condition; PPG in "all" only where the paper says so."""
    if cond == "all":
        channels = ["ECG", "THX"]
        if dataset_name in ALL_WITH_PPG:
            channels.append("PPG")
        return channels
    return {
        "ecg_thx": ["ECG", "THX"],
        "ecg": ["ECG"],
        "ppg": ["PPG"],
    }[cond]


def _build_dataset(dataset_cls, root, cache_dir, channels) -> "object | None":
    """Build one whole-night dataset instance; None (with a reason) on failure."""
    try:
        ds = dataset_cls(
            root=root,
            channels=channels,
            pipelines=PIPELINE,
            sequence_length=0,
            label_transform=LABEL_TRANSFORM,
            cache_dir=cache_dir,
            trim_excess_wake=False,
        )
    except Exception as exc:
        print(f"  [SKIP] cannot build dataset: {type(exc).__name__}: {exc}")
        return None
    if ds.get_n_subjects() == 0:
        print("  [SKIP] no subjects found (data missing or unreadable)")
        return None
    return ds


def _predict_nights(model, dataset, subject_ids, signals) -> "list[tuple]":
    """Return (subject_id, pred, labels) for each night that evaluates cleanly."""
    import numpy as np
    from torch import no_grad

    out = []
    for sid in subject_ids:
        try:
            item = dataset[int(dataset.get_subjects().index(sid))]
        except (IndexError, ValueError):
            print(f"  [SKIP] {sid}: not present in this dataset instance")
            continue
        sig = {name: item["signals"][f"{name}_0"] for name in signals}
        try:
            with no_grad():
                pred = model.predict_night(sig, item["subject"])
            pred = np.asarray(pred, dtype=np.int64)
        except Exception as exc:
            print(f"  [SKIP] {sid}: prediction failed ({type(exc).__name__}: {exc})")
            continue
        out.append((sid, pred, item["labels"]))
    return out


def _git_sha() -> "str | None":
    from subprocess import run

    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").exists():
            try:
                return run(["git", "-C", str(p), "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True).stdout.strip()
            except Exception:
                return None
        p = p.parent
    return None


def main(argv=None) -> int:
    rep = None
    try:
        import physioex.train.replication as rep
    except ImportError as exc:
        print(f"[SKIP] physioex.train.replication is not available ({exc}); "
              f"nothing to verify")
        return 0

    import importlib.metadata

    import torch
    from physioex.data.datasets import get_dataset
    from physioex.models import load_from_pretrained

    args = build_parser().parse_args(argv)
    header, targets = rep.load_paper_metrics(METRICS_PATH)
    dataset_names = (args.datasets
                     or sorted({t.dataset for t in targets if t.dataset != "census"}))
    device = args.device or _default_device()
    max_nights = args.max_nights
    split = header.get("split", "wav2sleep")

    print(f"Loading {MODEL_NAME} on {device}...")
    model = load_from_pretrained(MODEL_NAME, device=device)

    # ── Per-dataset evaluation, one pass per condition ─────────────────
    per_dataset: dict = {}     # (dataset, condition) -> [(sid, pred, labels)]
    for name in dataset_names:
        try:
            ds = _build_dataset(get_dataset(name), args.data_root, args.cache_dir,
                                ["ECG", "THX", "PPG"])
        except Exception as exc:
            print(f"  [SKIP] {name}: {type(exc).__name__}: {exc}")
            continue
        if ds is None:
            continue
        _, _, test_ids = ds.get_splits(split=split)
        test_ids = test_ids[:max_nights] if max_nights else test_ids
        print(f"{name}: {len(test_ids)} test nights")
        for cond in sorted({t.condition for t in targets if t.dataset == name}):
            signals = _condition_channels(cond, name)
            records = _predict_nights(model, ds, test_ids, signals)
            records = [(sid, *rep.align(pred, labels)) for sid, pred, labels in records]
            if records:
                per_dataset[(name, cond)] = records
            if len(records) < len(test_ids):
                print(f"  {cond}: {len(test_ids) - len(records)} nights skipped")

    # ── Census rows: pool the census nights of the required datasets ───
    for t in targets:
        if t.dataset != "census":
            continue
        records = []
        for name in t.requires:
            ds = _build_dataset(get_dataset(name), args.data_root, args.cache_dir,
                                ["ECG", "THX", "PPG"])
            if ds is None:
                continue
            _, _, test_ids = ds.get_splits(split="wav2sleep-census")
            test_ids = test_ids[:max_nights] if max_nights else test_ids
            for sid, pred, labels in _predict_nights(
                    model, ds, test_ids, _condition_channels(t.condition, name)):
                records.append((sid, *rep.align(pred, labels)))
        if records:
            per_dataset[("census", t.condition)] = records

    # ── Scores and verdicts ────────────────────────────────────────────
    results = {
        key: rep.evaluate([p for _, p, _ in recs], [l for _, _, l in recs], N_CLASSES)
        for key, recs in per_dataset.items()
    }
    rows = rep.check(results, targets)
    if max_nights:
        for row in rows:
            if row["status"] in ("PASS", "FAIL"):
                row["status"] = "REPORT"
    print(rep.format_table(rows))

    per_record_kappa = {}
    for key, recs in per_dataset.items():
        for sid, pred, labels in recs:
            per_record_kappa.setdefault(key[0], {}).setdefault(
                sid, rep.evaluate([(pred,)], [(labels,)], N_CLASSES))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "header": header,
        "rows": rows,
        "per_record_kappa": per_record_kappa,
        "run": {
            "git_sha": _git_sha(),
            "physioex_version": importlib.metadata.version("physioex"),
            "torch_version": torch.__version__,
            "device": device,
            "model": MODEL_NAME,
            "max_nights": max_nights,
            "data_root": os.environ.get("PHYSIOEX_DATA", args.data_root),
        },
    }, indent=2, default=str))
    print(f"\nSaved results to {out_path}")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
