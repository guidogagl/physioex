"""Evaluate protosleepnet-gagliardi on all supported datasets.

Downloads the pretrained model via ``load_from_pretrained("prosleepnet-gagliardi")``,
then evaluates it on the test split of every available dataset using per-subject
voting (sliding window L=21).

Usage:
    python examples/pretrained/protosleepnet-gagliardi/test_pretrained.py --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/test_pretrained.py --gpu_id 0 --datasets shhs sleepedf
    python examples/pretrained/protosleepnet-gagliardi/test_pretrained.py --gpu_id 0 --upload
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models import load_from_pretrained
from physioex.models.prosleepnet import ProtoSleepTransformerTrainer

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21
MODEL_NAME = "prosleepnet-gagliardi"


def evaluate_on_dataset(model, dataset_name, gpu_id=None):
    """Evaluate model on a single dataset. Returns metrics dict or None on failure."""
    try:
        DatasetClass = get_dataset(dataset_name)
        dataset = DatasetClass(
            channels=CHANNELS,
            pipelines=PIPELINE,
            sequence_length=SEQ_LEN,
        )
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: cannot load dataset ({e})")
        return None

    n_subjects = dataset.get_n_subjects()
    if n_subjects == 0:
        print(f"  [SKIP] {dataset_name}: no subjects found")
        return None

    _, _, test_ids = dataset.get_splits(fold=0)
    print(f"  {dataset_name}: {n_subjects} subjects, {len(test_ids)} in test split")

    try:
        results = ProtoSleepTransformerTrainer.voting_evaluate(
            model=model,
            dataset=dataset,
            L=SEQ_LEN,
            fold=0,
            gpu_id=gpu_id,
        )
    except Exception as e:
        print(f"  [FAIL] {dataset_name}: evaluation error ({e})")
        return None

    serializable = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return serializable


def print_summary_table(all_results):
    """Print a formatted table of per-dataset metrics."""
    header = (
        f"{'Dataset':15s} {'Acc':>7s} {'F1':>7s} {'Kappa':>7s} {'Prec':>7s} {'Rec':>7s}"
    )
    print("\n" + "=" * 60)
    print("ProtoSleepTransformer-Gagliardi — Per-Dataset Evaluation Summary")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for name, metrics in sorted(all_results.items()):
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_score", 0)
        kap = metrics.get("cohen_kappa", 0)
        pre = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        print(f"{name:15s} {acc:7.4f} {f1:7.4f} {kap:7.4f} {pre:7.4f} {rec:7.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protosleepnet-gagliardi on all supported datasets"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="GPU device id (None for CPU)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to evaluate (default: all available)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save metrics.json",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload metrics.json to HuggingFace Hub",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pretrained ProtoSleepTransformer from HuggingFace...")
    model = load_from_pretrained(MODEL_NAME)
    print(
        f"Model: {type(model).__name__}, "
        f"params={sum(p.numel() for p in model.parameters())}\n"
    )

    dataset_names = args.datasets if args.datasets else available_datasets()

    all_results = {}
    for name in dataset_names:
        print(f"Evaluating on {name}...")
        metrics = evaluate_on_dataset(model, name, gpu_id=args.gpu_id)
        if metrics is not None:
            all_results[name] = metrics
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            kap = metrics.get("cohen_kappa", 0)
            print(f"  -> acc={acc:.4f}, f1={f1:.4f}, kappa={kap:.4f}\n")

    print_summary_table(all_results)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved per-dataset metrics to {metrics_path}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo=f"{MODEL_NAME}/metrics.json",
            repo_id="4rooms/physioex",
            repo_type="model",
        )
        print(f"Uploaded metrics.json to 4rooms/physioex/{MODEL_NAME}/")


if __name__ == "__main__":
    main()
