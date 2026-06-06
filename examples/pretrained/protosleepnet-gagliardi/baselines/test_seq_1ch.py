"""Evaluate single-channel SeqSleepNet-Phan on MASS (in-domain).

Loads the pretrained model from HuggingFace and evaluates on the MASS
test split using per-subject voting (L=20). Saves metrics.json.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_seq_1ch.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.data.multi import MultiDataset
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer

MODEL_NAME = "seqsleepnet-phan"
CHANNELS = ["EEG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 20


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SeqSleepNet-Phan 1ch on MASS"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/seqsleepnet-phan-1ch-eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model from HF ──────────────────────────────────────────
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = load_from_pretrained(MODEL_NAME)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}, params: {n_params:,}")

    # ── Dataset: MASS all cohorts ────────────────────────────────────
    MASS = get_dataset("mass")
    cohort_datasets = []
    for cohort in [1, 2, 3, 4, 5]:
        ds = MASS(
            cohort=cohort,
            channels=CHANNELS,
            pipelines=PIPELINE,
            sequence_length=SEQ_LEN,
        )
        n = ds.get_n_subjects()
        print(f"  MASS SS{cohort:02d}: {n} subjects")
        if n > 0:
            cohort_datasets.append(ds)

    dataset = MultiDataset(cohort_datasets)
    print(f"Combined: {dataset}")

    # ── Voting evaluation on test split ──────────────────────────────
    results = Trainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=SEQ_LEN,
        fold=0,
        gpu_id=args.gpu_id,
    )

    # ── Save metrics ─────────────────────────────────────────────────
    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults: accuracy={results['accuracy']:.4f}, "
          f"f1={results['f1_score']:.4f}, kappa={results['cohen_kappa']:.4f}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
