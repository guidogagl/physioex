"""Extract pre-sequence epoch embeddings h(x) from baseline models.

For each baseline model (sleeptransformer-gagliardi, seqsleepnet-gagliardi),
extracts the epoch encoder output (before the sequence encoder) for every
epoch in the SHHS train and test splits.

The baselines are ProtoSleepTransformer / ProtoSeqSleepNet with
use_channel_mixer=False, use_prototypes=False (per-channel encoding + mean pool).

h(x) pipeline:
  (B, L, C, T, F) → per-channel epoch encoder → (B*L, C, d_model) → mean → (B*L, d_model)

Each epoch maps to exactly one embedding vector (no sequence context).

Usage:
    python examples/pretrained/protosleepnet-gagliardi/extract_epoch_embeddings.py \
        --model_dir /path/to/pretrained/st-baseline \
        --output_dir /path/to/save \
        --gpu_id 0
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.train.trainer import Trainer

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21


def load_model(model_dir, device):
    """Load model from config.json + model.pt in a local directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    module_path, class_name = config["model_class"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    model = ModelClass(**config["model_kwargs"])
    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


@torch.no_grad()
def extract_epoch_encoder(model, x):
    """Extract epoch-level embeddings h(x) from the epoch encoder.

    Handles both ProtoSleepTransformer and ProtoSeqSleepNet (per-channel
    encoding + mean pool), as well as plain SleepTransformer / SeqSleepNet.

    Args:
        model: A model instance (any variant).
        x: (N, C, T, F) input spectrograms (N epochs, C channels).

    Returns:
        (N, d_model) epoch embeddings (mean-pooled across channels).
    """
    N, C, T, F = x.shape

    # ProtoSleepTransformer / ProtoSeqSleepNet: per-channel epoch encoder
    if hasattr(model, "epoch_encoder") and hasattr(model, "in_chan"):
        x_flat = x.reshape(N * C, 1, T, F)
        embs = model.epoch_encoder(x_flat)   # (N*C, d_model)
        embs = embs.reshape(N, C, -1)        # (N, C, d_model)
        return embs.mean(dim=1)              # (N, d_model)

    # Plain SleepTransformer: epoch_encoder expects (B, C, T, F) → (B, d_model)
    if hasattr(model, "epoch_encoder"):
        return model.epoch_encoder(x)        # (N, d_model)

    # Plain SeqSleepNet: filterbank → seqn1 → attention
    if hasattr(model, "filterbank") and hasattr(model, "seqn1"):
        z = model.filterbank(x)              # (N, C, T, D)
        z = z.permute(0, 2, 1, 3)
        N2, T2, C2, D = z.shape
        z = z.reshape(N2, T2, C2 * D)
        z, _ = model.seqn1(z)
        z = model.attention(z)               # (N, hidden)
        return z

    raise ValueError(f"Unknown model type: {type(model).__name__}")


def extract_split(model, dataloader, device, batch_size=256):
    """Extract epoch embeddings for all subjects in a dataloader.

    Args:
        model: Frozen model in eval mode.
        dataloader: DataLoader yielding full-night recordings (batch_size=1).
        device: Torch device.
        batch_size: Number of epochs per forward pass.

    Returns:
        Z: (N_total, d_model) embeddings
        Y: (N_total,) labels
    """
    all_embs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting"):
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch)    # (1, night_len, C, T, F)
            targets = batch["labels"]         # (1, night_len)
        else:
            inputs, targets = batch

        # Flatten to per-epoch: (night_len, C, T, F)
        x = inputs.squeeze(0).to(device)
        y = targets.squeeze(0).numpy()

        # Process in batches to avoid OOM
        N = x.shape[0]
        embs = []
        for i in range(0, N, batch_size):
            chunk = x[i : i + batch_size]
            e = extract_epoch_encoder(model, chunk)
            embs.append(e.cpu().numpy())

        embs = np.concatenate(embs, axis=0)  # (night_len, d_model)
        all_embs.append(embs)
        all_labels.append(y)

    Z = np.concatenate(all_embs, axis=0).astype(np.float32)
    Y = np.concatenate(all_labels, axis=0).astype(np.int64)

    return Z, Y


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-sequence epoch embeddings from baseline models"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with config.json + model.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load model
    model_name = os.path.basename(args.model_dir)
    print(f"Loading model: {model_name} from {args.model_dir}")
    model, config = load_model(args.model_dir, device)
    print(f"  Class: {config['model_class']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    ds_kwargs = {"visit": 1} if args.dataset == "shhs" else {}
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=SEQ_LEN,
        **ds_kwargs,
    )

    # Build dataloaders (batch_size=1, one subject per batch)
    train_loader, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        fold=args.fold,
    )

    # Output directory
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Extract train embeddings
    print(f"\nExtracting train embeddings ({len(train_loader)} subjects)...")
    Z_train, Y_train = extract_split(model, train_loader, device, args.batch_size)
    print(f"  Train: {Z_train.shape[0]} epochs, d_model={Z_train.shape[1]}")

    train_valid = Y_train >= 0
    print(f"  Valid epochs: {train_valid.sum()} / {len(Y_train)}")

    np.save(os.path.join(out_dir, "train_embeddings.npy"), Z_train)
    np.save(os.path.join(out_dir, "train_labels.npy"), Y_train)

    # Extract test embeddings
    print(f"\nExtracting test embeddings ({len(test_loader)} subjects)...")
    Z_test, Y_test = extract_split(model, test_loader, device, args.batch_size)
    print(f"  Test: {Z_test.shape[0]} epochs, d_model={Z_test.shape[1]}")

    test_valid = Y_test >= 0
    print(f"  Valid epochs: {test_valid.sum()} / {len(Y_test)}")

    np.save(os.path.join(out_dir, "test_embeddings.npy"), Z_test)
    np.save(os.path.join(out_dir, "test_labels.npy"), Y_test)

    print(f"\nSaved to {out_dir}/")
    print(f"  train_embeddings.npy: {Z_train.shape}")
    print(f"  train_labels.npy:     {Y_train.shape}")
    print(f"  test_embeddings.npy:  {Z_test.shape}")
    print(f"  test_labels.npy:      {Y_test.shape}")


if __name__ == "__main__":
    main()
