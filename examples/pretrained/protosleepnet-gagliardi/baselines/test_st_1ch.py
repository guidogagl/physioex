"""Evaluate single-channel SleepTransformer-Phan on SHHS (in-domain).

Loads the pretrained model from HuggingFace and evaluates on the SHHS
test split using per-subject voting (L=21). Saves metrics.json.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_st_1ch.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer

MODEL_NAME = "sleeptransformer-phan"
CHANNELS = ["EEG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SleepTransformer-Phan 1ch on SHHS"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/sleeptransformer-phan-1ch-eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model from HF ──────────────────────────────────────────
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = load_from_pretrained(MODEL_NAME)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}, params: {n_params:,}")

    # ── Dataset: SHHS visit 1 ────────────────────────────────────────
    DatasetClass = get_dataset("shhs")
    dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=SEQ_LEN,
        visit=1,
    )
    print(f"SHHS: {dataset.get_n_subjects()} subjects")

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
