"""Verify Jones, Itti & Sheth (2024) ECG-only sleep staging against the paper.

Loads ``load_from_pretrained("jones-ecg-<variant>")`` (default: the paper's
primary checkpoint) and evaluates it on the census-balanced 500-night test
set of Jones et al. (2024) — the ``wav2sleep-census`` split, spread over
ccshs, cfs, chat, mesa and wsc — and on the per-dataset test sets.  The
headline is the **median per-recording kappa** (0.725 in the paper);
``reference_per_record.csv`` ships the authors' own per-recording kappas, and
the median over exactly the recordings we evaluated is printed next to ours
(the like-for-like comparison when CFS/CHAT are missing locally).

Usage (on the cluster where the NSRR data live):
    python examples/pretrained/jones-ecg/verify.py \
        --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
    python examples/pretrained/jones-ecg/verify.py --variant realtime
    python examples/pretrained/jones-ecg/verify.py --max-nights 2   # smoke
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

PIPELINE = "jones_ecg"
N_CLASSES = 5
MODEL_NAMES = ("primary", "realtime", "no-demographics", "no-time",
               "only-ecg", "cyclic", "cyclic-realtime")
METRICS_PATH = Path(__file__).parent / "paper_metrics.json"
REFERENCE_CSV = Path(__file__).parent / "reference_per_record.csv"
SPLIT = "wav2sleep-census"
CENSUS_DATASETS = ("ccshs", "cfs", "chat", "mesa", "wsc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the Jones et al. (2024) ECG model on the census test set")
    parser.add_argument("--data-root", default=None,
                        help="raw data root (default: $PHYSIOEX_DATA)")
    parser.add_argument("--cache-dir", default=None,
                        help="preprocessing cache (default: $PHYSIOEX_CACHE_DIR)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="datasets to evaluate (default: the census datasets)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--variant", default="primary", choices=MODEL_NAMES,
                        help="which released checkpoint (default: primary)")
    parser.add_argument("--max-nights", type=int, default=None,
                        help="smoke run: at most N nights per dataset "
                             "(PASS/FAIL reported as REPORT)")
    parser.add_argument("--out", default="./verify_results.json",
                        help="path of the output JSON")
    return parser


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_dataset(dataset_cls, root, cache_dir, **extra):
    """Build one dataset (whole-night ECG, jones preset); None on failure."""
    try:
        ds = dataset_cls(
            root=root,
            channels=["ECG"],
            pipelines=PIPELINE,
            **extra,
            sequence_length=0,
            # Whole nights are read once each: keep few files open (a large
            # memmap LRU can stall a GPFS client on the cluster node).
            memmap_cache_size=4,
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


def _reference_kappas() -> dict:
    """Map session_id -> o_kappa from the authors' per-recording CSV."""
    out = {}
    with open(REFERENCE_CSV) as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row["o_kappa"])
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
    import numpy as np
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models import load_from_pretrained

    args = build_parser().parse_args(argv)
    header, targets = rep.load_paper_metrics(METRICS_PATH)
    dataset_names = args.datasets or list(CENSUS_DATASETS)
    device = args.device or _default_device()
    max_nights = args.max_nights
    model_name = f"jones-ecg-{args.variant}"

    print(f"Loading {model_name} on {device}...")
    model = load_from_pretrained(model_name, device=device)
    reference = _reference_kappas()

    # ── Per-dataset evaluation on the census nights ────────────────────
    # Each (dataset, condition) gets its own record list; the census row
    # pools all of them.  Session ids are mapped to the NSRR ids used in
    # reference_per_record.csv via dataset.session_id().
    per_dataset: dict = {}   # (dataset, condition) -> [(sid, pred, labels)]
    session_id_of: dict = {}
    for name, extra in [(n, kw) for n in dataset_names for kw in rep.replication_kwargs(n)]:
        try:
            ds = _build_dataset(get_dataset(name), args.data_root, args.cache_dir, **extra)
        except Exception as exc:
            print(f"  [SKIP] {name}: {type(exc).__name__}: {exc}")
            continue
        if ds is None:
            continue
        _, _, test_ids = ds.get_splits(split=SPLIT)
        test_ids = test_ids[:max_nights] if max_nights else test_ids
        print(f"{name} {extra}: {len(test_ids)} census test nights")
        present = set(ds.get_subjects())
        records = []
        for sid, signals, labels, meta in rep.iter_nights(ds, [s for s in test_ids if s in present]):
            if "ECG" not in signals:
                continue
            pred, lab, _ = rep.align(model.predict_night({"ECG": signals["ECG"]}, meta), labels)
            records.append((ds.session_id(sid), pred, lab))
            session_id_of[sid] = ds.session_id(sid)
        if records:
            per_dataset.setdefault((name, "ecg"), []).extend(records)
        if len(records) < len(test_ids):
            print(f"  ecg: {len(test_ids) - len(records)} nights skipped")

    census_records = [rec for recs in per_dataset.values() for rec in recs]
    if census_records:
        per_dataset[("census", "ecg")] = census_records

    results = {
        key: rep.evaluate([l for _, _, l in recs], [p for _, p, _ in recs], N_CLASSES)
        for key, recs in per_dataset.items()
    }
    rows = rep.check(results, targets)
    if max_nights:
        for row in rows:
            if row["status"] in ("PASS", "FAIL"):
                row["status"] = "REPORT"
    print(rep.format_table(rows))

    # ── Like-for-like reference: median o_kappa over the same recordings ─
    ref_median = {}
    for key, recs in per_dataset.items():
        ref = [reference[sid] for sid, _, _ in recs if sid in reference]
        ref_median[key[0]] = float(np.median(ref)) if ref else None
        shown = f"{ref_median[key[0]]:.4f}" if ref else "n/a"
        print(f"  reference median o_kappa over the same {key[0]} recordings: {shown}")

    per_record_kappa = {
        key[0]: dict(zip([sid for sid, _, _ in recs], results[key]["per_record_kappa"]))
        for key, recs in per_dataset.items()
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "header": header,
        "rows": rows,
        "per_record_kappa": per_record_kappa,
        "reference_median_over_same_recordings": ref_median,
        "run": {
            "git_sha": _git_sha(),
            "physioex_version": importlib.metadata.version("physioex"),
            "torch_version": torch.__version__,
            "device": device,
            "model": model_name,
            "variant": args.variant,
            "max_nights": max_nights,
            "data_root": os.environ.get("PHYSIOEX_DATA", args.data_root),
        },
    }, indent=2, default=str))
    print(f"\nSaved results to {out_path}")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
