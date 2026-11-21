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
        channels = ["ABD", "THX", "ECG"]
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

    # ── One pass over the nights; every condition predicted per night ──
    census = {t.condition: t.requires for t in targets if t.dataset == "census"}
    census_names = set().union(*census.values()) if census else set()
    names = sorted(set(dataset_names) | census_names)
    per_dataset: dict = {}     # (dataset, condition) -> [(sid, pred, labels)]
    for name in names:
        ds = _build_dataset(get_dataset(name), args.data_root, args.cache_dir,
                            ["ECG", "THX", "ABD", "PPG"])
        if ds is None:
            continue
        own = {t.condition for t in targets if t.dataset == name} if name in dataset_names else set()
        test_ids = list(ds.get_splits(split=split)[2]) if own else []
        cen_conds = {c for c, req in census.items() if name in req}
        cen_ids = list(ds.get_splits(split="wav2sleep-census")[2]) if cen_conds else []
        if max_nights:
            test_ids, cen_ids = test_ids[:max_nights], cen_ids[:max_nights]
        wanted = list(dict.fromkeys(test_ids + cen_ids))
        present = set(ds.get_subjects())
        missing = [s for s in wanted if s not in present]
        if missing:
            print(f"  {name}: {len(missing)} split nights not in the dataset, skipped")
        print(f"{name}: {len(test_ids)} test nights, {len(cen_ids)} census nights")
        test_set, cen_set = set(test_ids), set(cen_ids)
        for sid, signals, labels, _meta in rep.iter_nights(ds, [s for s in wanted if s in present]):
            for cond in own | cen_conds:
                sig = {k: v for k, v in signals.items() if k in _condition_channels(cond, name)}
                if not sig:
                    continue
                pred, lab, _ = rep.align(model.predict_night(sig), labels)
                if cond in own and sid in test_set:
                    per_dataset.setdefault((name, cond), []).append((sid, pred, lab))
                if cond in cen_conds and sid in cen_set:
                    per_dataset.setdefault(("census", cond), []).append((sid, pred, lab))

    # ── Scores and verdicts ────────────────────────────────────────────
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

    per_record_kappa = {
        f"{key[0]}/{key[1]}": dict(zip([sid for sid, _, _ in recs],
                                       results[key]["per_record_kappa"]))
        for key, recs in per_dataset.items()
    }

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
