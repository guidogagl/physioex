"""Verify SleepPPG-Net (third-party replica) against the published number.

Loads ``load_from_pretrained("sleepppgnet-replica")`` — the community replica
weights (``sleep-is-all-you-need/SPPPPG-Net``, fair non-commercial research
license) of Kotzen et al.'s (2023) SleepPPG-Net — and evaluates it on the 204
MESA test recordings of the replica's split
(``split="sleepppgnet-replica"``).  The paper's headline is the median
per-patient kappa (0.75, IQR 0.69-0.81); it is reported next to our value,
not asserted (``assert: false`` in ``paper_metrics.json``), because the
weights are a third-party re-training, not the authors' model.

Usage (on the cluster where the MESA data live):
    python examples/pretrained/sleepppgnet/verify.py \
        --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
    python examples/pretrained/sleepppgnet/verify.py --max-nights 2   # smoke
"""
import argparse
import json
import os
import sys
from pathlib import Path

MODEL_NAME = "sleepppgnet-replica"
PIPELINE = "sleepppgnet"
LABEL_TRANSFORM = "aasm4"
N_CLASSES = 4
METRICS_PATH = Path(__file__).parent / "paper_metrics.json"
DATASET = "mesa"
SPLIT = "sleepppgnet-replica"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the SleepPPG-Net replica on the MESA test split")
    parser.add_argument("--data-root", default=None,
                        help="raw data root (default: $PHYSIOEX_DATA)")
    parser.add_argument("--cache-dir", default=None,
                        help="preprocessing cache (default: $PHYSIOEX_CACHE_DIR)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="datasets to evaluate (default: mesa)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--max-nights", type=int, default=None,
                        help="smoke run: at most N nights "
                             "(PASS/FAIL reported as REPORT)")
    parser.add_argument("--out", default="./verify_results.json",
                        help="path of the output JSON")
    return parser


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_dataset(dataset_cls, root, cache_dir):
    """Build the MESA dataset (whole-night PPG); None (with a reason) on failure."""
    try:
        ds = dataset_cls(
            root=root,
            channels=["PPG"],
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
        print("  [SKIP] no subjects found (MESA data missing or unreadable)")
        return None
    return ds


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
    dataset_names = args.datasets or [DATASET]
    device = args.device or _default_device()
    max_nights = args.max_nights

    print(f"Loading {MODEL_NAME} on {device}...")
    model = load_from_pretrained(MODEL_NAME, device=device)

    per_dataset: dict = {}    # (dataset, condition) -> [(sid, pred, labels)]
    for name in dataset_names:
        try:
            ds = _build_dataset(get_dataset(name), args.data_root, args.cache_dir)
        except Exception as exc:
            print(f"  [SKIP] {name}: {type(exc).__name__}: {exc}")
            continue
        if ds is None:
            continue
        _, _, test_ids = ds.get_splits(split=SPLIT)
        test_ids = test_ids[:max_nights] if max_nights else test_ids
        print(f"{name}: {len(test_ids)} test nights")
        cond = "ppg"
        records = []
        for sid in test_ids:
            try:
                item = ds[int(ds.get_subjects().index(sid))]
            except (IndexError, ValueError):
                print(f"  [SKIP] {sid}: not present in this dataset instance")
                continue
            from torch import no_grad

            try:
                with no_grad():
                    pred = model.predict_night(
                        {"PPG": item["signals"]["PPG_0"]}, item["subject"])
            except Exception as exc:
                print(f"  [SKIP] {sid}: prediction failed ({type(exc).__name__}: {exc})")
                continue
            pred, labels, _ = rep.align(pred, item["labels"])
            records.append((sid, pred, labels))
        if records:
            per_dataset[(name, cond)] = records
        if len(records) < len(test_ids):
            print(f"  {cond}: {len(test_ids) - len(records)} nights skipped")

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
