"""Evaluate seqsleepnet-phan on each STAGES site individually.

Loads the pretrained model via ``load_from_pretrained("seqsleepnet-phan")``,
then evaluates it on the test split of each of the 13 STAGES clinical sites
using per-subject voting (sliding window L=20).

Results are saved to ``stages_sites_metrics.json`` and a summary table is
printed to stdout.

Usage:
    python examples/pretrained/seqsleepnet-phan/test_stages_sites.py --gpu_id 0
    python examples/pretrained/seqsleepnet-phan/test_stages_sites.py --gpu_id 0 --sites BOGN GSBB
"""
import argparse
import json
import os
import time
import traceback

import torch

from physioex.data.datasets.stages import STAGESDataset, SITES
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer

CHANNELS = ["EEG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 20
MODEL_NAME = "seqsleepnet-phan"


def evaluate_on_site(model, site, gpu_id=None):
    """Evaluate model on a single STAGES site. Returns (metrics_dict, elapsed) or (None, elapsed)."""
    t0 = time.time()
    try:
        dataset = STAGESDataset(
            site=site,
            channels=CHANNELS,
            pipelines=PIPELINE,
            sequence_length=SEQ_LEN,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [SKIP] {site}: cannot load dataset ({e})")
        traceback.print_exc()
        return None, elapsed

    n_subjects = dataset.get_n_subjects()
    if n_subjects == 0:
        elapsed = time.time() - t0
        print(f"  [SKIP] {site}: no subjects found")
        return None, elapsed

    _, _, test_ids = dataset.get_splits(fold=0)
    print(f"  {site}: {n_subjects} subjects total, {len(test_ids)} in test split")

    try:
        results = Trainer.voting_evaluate(
            model=model,
            dataset=dataset,
            L=SEQ_LEN,
            fold=0,
            gpu_id=gpu_id,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {site}: evaluation error ({e})")
        traceback.print_exc()
        return None, elapsed

    elapsed = time.time() - t0

    serializable = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    serializable["n_subjects"] = n_subjects
    serializable["n_test_subjects"] = len(test_ids)
    serializable["elapsed_sec"] = round(elapsed, 1)

    return serializable, elapsed


def print_summary_table(all_results):
    """Print a formatted table of per-site metrics."""
    header = f"{'Site':6s} {'N_test':>6s} {'Acc':>7s} {'F1':>7s} {'Kappa':>7s} {'Prec':>7s} {'Rec':>7s} {'Time(s)':>8s}"
    print("\n" + "=" * 70)
    print("SeqSleepNet-Phan — Per-Site STAGES Evaluation")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for name, metrics in sorted(all_results.items()):
        n_test = metrics.get("n_test_subjects", "?")
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_score", 0)
        kap = metrics.get("cohen_kappa", 0)
        pre = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        elapsed = metrics.get("elapsed_sec", 0)
        print(f"{name:6s} {n_test:>6} {acc:7.4f} {f1:7.4f} {kap:7.4f} {pre:7.4f} {rec:7.4f} {elapsed:8.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate seqsleepnet-phan on each STAGES site individually"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="GPU device id (None for CPU)"
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help=f"Specific sites to evaluate (default: all {len(SITES)}). Choices: {SITES}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stages_sites_metrics.json",
        help="Output JSON path (default: stages_sites_metrics.json)",
    )
    args = parser.parse_args()

    sites = args.sites if args.sites else SITES
    for s in sites:
        if s not in SITES:
            parser.error(f"Unknown site {s!r}. Must be one of {SITES}")

    print("Loading pretrained SeqSleepNet from HuggingFace...")
    model = load_from_pretrained(MODEL_NAME)
    print(
        f"Model: {type(model).__name__}, "
        f"params={sum(p.numel() for p in model.parameters())}\n"
    )

    all_results = {}
    failed_sites = []
    for site in sites:
        print(f"Evaluating on {site}...")
        metrics, elapsed = evaluate_on_site(model, site, gpu_id=args.gpu_id)
        if metrics is not None:
            all_results[site] = metrics
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            kap = metrics.get("cohen_kappa", 0)
            print(f"  -> acc={acc:.4f}, f1={f1:.4f}, kappa={kap:.4f} ({elapsed:.0f}s)\n")
        else:
            failed_sites.append(site)

    print_summary_table(all_results)

    if failed_sites:
        print(f"\nFailed/skipped sites: {failed_sites}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved per-site metrics to {args.output}")


if __name__ == "__main__":
    main()
