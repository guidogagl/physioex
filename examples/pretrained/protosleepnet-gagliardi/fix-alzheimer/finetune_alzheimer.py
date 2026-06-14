"""Fine-tune ProtoSleepNet epoch encoder on Alzheimer HC dataset.

Freezes everything except the epoch encoder, trains with low LR on
Healthy Controls, evaluates on HC test + AD.

Usage:
    python finetune_alzheimer.py --backbone seq --checkpoint /path/to/model.pt --gpu_id 0
    python finetune_alzheimer.py --backbone st --checkpoint /path/to/model.pt --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.protosleepnet import ProtoSleepNet, ProtoSleepNetTrainer

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"

BACKBONE_CONFIG = {
    "seq": {"factory": "from_seq_sleep_net", "seq_len": 20},
    "st": {"factory": "from_sleep_transformer", "seq_len": 21},
}

MIXER_KWARGS = {
    "use_channel_mixer": True,
    "cdropout": 0.5,
    "cm_n_heads": 4,
    "cm_d_ff": 256,
    "cm_n_layers": 1,
}


def load_model(backbone, checkpoint_path, device):
    factory = getattr(ProtoSleepNet, BACKBONE_CONFIG[backbone]["factory"])
    model = factory(n_channels=3, **MIXER_KWARGS)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ProtoSleepNet epoch encoder on Alzheimer HC"
    )
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train from random init (no checkpoint, no freeze)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_alzheimer_output")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seq_len = BACKBONE_CONFIG[args.backbone]["seq_len"]

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ── Load model ───────────────────────────────────────────────
    if args.from_scratch:
        print(f"Training ProtoSleepNet ({args.backbone}) from scratch")
        factory = getattr(ProtoSleepNet, BACKBONE_CONFIG[args.backbone]["factory"])
        model = factory(n_channels=3, **MIXER_KWARGS).to(device)
        n_trainable = sum(p.numel() for p in model.parameters())
        print(f"Params (all trainable): {n_trainable:,}")
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --from_scratch is set")
        print(f"Loading ProtoSleepNet ({args.backbone}) from {args.checkpoint}")
        model = load_model(args.backbone, args.checkpoint, device)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

        # Freeze all, unfreeze epoch encoder only
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.epoch_encoder.parameters():
            p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable (epoch encoder): {n_trainable:,}")
        print(f"Frozen: {n_frozen:,}")

    # ── Dataset: HC ──────────────────────────────────────────────
    DatasetClass = get_dataset("alzheimers")
    hc_dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=seq_len,
        subset="HC",
    )
    print(f"HC dataset: {hc_dataset.get_n_subjects()} subjects")

    # ── Train ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fine-tuning epoch encoder on HC (lr={args.lr})")
    print(f"{'='*60}")

    model = ProtoSleepNetTrainer.train(
        model=model,
        dataset=hc_dataset,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=0,
        train_batch_size=args.batch_size,
        fold=0,
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=args.patience,
    )

    # ── Save model ───────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # ── Evaluate on HC test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating on HC test split")
    print(f"{'='*60}")

    hc_results = ProtoSleepNetTrainer.voting_evaluate(
        model=model,
        dataset=hc_dataset,
        L=seq_len,
        fold=0,
        gpu_id=args.gpu_id,
    )
    hc_metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in hc_results.items()}
    print(f"HC test: acc={hc_results['accuracy']:.4f}, "
          f"f1={hc_results['f1_score']:.4f}, kappa={hc_results['cohen_kappa']:.4f}")

    # ── Evaluate on AD ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating on AD (all subjects)")
    print(f"{'='*60}")

    ad_dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=seq_len,
        subset="AD",
    )
    ad_results = ProtoSleepNetTrainer.voting_evaluate(
        model=model,
        dataset=ad_dataset,
        L=seq_len,
        fold=0,
        gpu_id=args.gpu_id,
    )
    ad_metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in ad_results.items()}
    print(f"AD: acc={ad_results['accuracy']:.4f}, "
          f"f1={ad_results['f1_score']:.4f}, kappa={ad_results['cohen_kappa']:.4f}")

    # ── Save results ─────────────────────────────────────────────
    results = {
        "backbone": args.backbone,
        "lr": args.lr,
        "method": "finetune_epoch_encoder",
        "hc_test": hc_metrics,
        "ad": ad_metrics,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
