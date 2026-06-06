"""Channel occlusion test for ChannelMixerWrapper models.

Loads the mixer model from a checkpoint, runs occlusion scenarios.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_occlusion_mixer.py \
        --checkpoint /path/to/checkpoint.pt --scenario clean --gpu_id 0
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from test_occlusion import (
    ChannelOcclusionWrapper,
    SCENARIOS,
    CLASS_NAMES,
    evaluate_subject,
    compute_metrics,
    build_dataset,
)
from train_seq_3ch_mixer import build_model as build_seq_mixer
from train_st_3ch_mixer import build_model as build_st_mixer
from physioex.train.trainer import Trainer

BUILDERS = {
    "seq": build_seq_mixer,
    "st": build_st_mixer,
}


def main():
    parser = argparse.ArgumentParser(description="Occlusion test for mixer models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="seq", choices=["seq", "st"])
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cdropout", type=float, default=0.5)
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(SCENARIOS.keys()),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load model
    model = BUILDERS[args.backbone](cdropout=args.cdropout)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    channels = ["EEG", "EOG", "EMG"]
    dataset = build_dataset(args.dataset, channels, "seqsleepnet", args.seq_len)

    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=0,
    )
    print(f"Dataset: {args.dataset}, seq_len: {args.seq_len}, test subjects: {len(test_loader)}")

    # Run scenarios
    scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS

    all_metrics = {}
    for scenario_name, scenario_cfg in scenarios_to_run.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        wrapped = ChannelOcclusionWrapper(
            model,
            mode=scenario_cfg.get("mode"),
            p=scenario_cfg.get("p", 0.0),
            channels_to_occlude=scenario_cfg.get("channels_to_occlude"),
        )

        subject_predictions = []
        all_proba = []
        all_targets = []

        for subj_idx, batch in enumerate(tqdm(test_loader, desc=scenario_name)):
            if isinstance(batch, dict) and "signals" in batch:
                from physioex.data.collate import stack_channels
                inputs = stack_channels(batch)
                targets = batch["labels"]
            else:
                inputs, targets = batch

            proba = evaluate_subject(wrapped, inputs, args.seq_len, device)
            targets_flat = targets.reshape(-1)
            subject_predictions.append({
                "subject_idx": subj_idx,
                "proba": proba.tolist(),
                "labels": targets_flat.tolist(),
            })
            all_proba.append(proba)
            all_targets.append(targets_flat)

        metrics = compute_metrics(all_proba, all_targets)
        all_metrics[scenario_name] = metrics

        print(
            f"  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  "
            f"kappa={metrics['cohen_kappa']:.4f}"
        )
        print(
            f"  per-class F1: "
            + "  ".join(f"{k}={v:.3f}" for k, v in metrics["f1_per_class"].items())
        )

        # Save per-scenario
        suffix = f"_{scenario_name}" if args.scenario else ""
        pred_path = os.path.join(args.output_dir, f"occlusion_predictions_{scenario_name}.json")
        with open(pred_path, "w") as f:
            json.dump(subject_predictions, f)

        metrics_path = os.path.join(args.output_dir, f"occlusion_metrics_{scenario_name}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary
    if len(all_metrics) > 1 and "clean" in all_metrics:
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        for sn, m in all_metrics.items():
            row = f"{sn:12s}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['cohen_kappa']:.4f}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
