"""Pretrain SleepTokenizer on multi-dataset schema.

Trains the epoch-level tokenizer with prototypical classification
and per-channel auxiliary loss. Uses monkey-patched Trainer._step
to handle dict-output models and modality-aware processing.

Training schema defined in: docs/training_schema.md

Usage:
    python examples/pretrained/protosleepnet-gagliardi/pretrain.py --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/pretrain.py --gpu_id 0 --data_root /path/to/data
"""
import argparse
import json
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.data.datasets import (
    SHHSDataset,
    MESADataset,
    STAGESDataset,
    MrOSDataset,
    WSCDataset,
    ParkinsonsDataset,
    AlzheimersDataset,
    SleepEDFDataset,
    HMCDataset,
    HPAPDataset,
)
from physioex.data.multi import MultiDataset
from physioex.models.sleep_tokenizer import (
    SleepTokenizer,
    build_modality_ids,
    N_MODALITY_TYPES,
)
from physioex.train.trainer import Trainer
from physioex.train.metrics import accuracy_score

MODEL_NAME = "protosleepnet-gagliardi"

MODEL_KWARGS = {
    "n_classes": 5,
    "d_model": 128,
    "unet_depth": 5,
    "unet_filters": [1, 64, 128, 256, 256, 256],
    "unet_k": 9,
    "unet_decode_levels": 3,
    "n_sab_layers": 2,
    "n_heads": 4,
    "ff_dim": 256,
    "n_seeds": 4,
    "tau": 0.1,
    "ema_alpha": 0.99,
    "p_batch_dropout": 0.3,
    "p_modality_dropout": 0.3,
}

# Default data root: environment variable or fallback path
DATA_ROOT_DEFAULT = os.environ.get(
    "SLEEP_DATA_ROOT",
    "/home/dev/sleep-data/raw-sleep/"
)

# Training datasets configuration
# Based on docs/training_schema.md
TRAINING_DATASETS = {
    "shhs": {
        "class": SHHSDataset,
        "kwargs": {"visit": 1},
        "name": "SHHS Visit 1",
    },
    "mesa": {
        "class": MESADataset,
        "kwargs": {},
        "name": "MESA",
    },
    "stages": {
        "class": STAGESDataset,
        "kwargs": {},
        "name": "STAGES",
    },
    "mros": {
        "class": MrOSDataset,
        "kwargs": {},
        "name": "MrOS",
    },
    "wsc": {
        "class": WSCDataset,
        "kwargs": {"visit": 1},
        "name": "WSC Visit 1",
    },
    "parkinsons": {
        "class": ParkinsonsDataset,
        "kwargs": {"recording": "night", "group": "HOA"},
        "name": "Parkinson's (HOA night)",
    },
    "alzheimers": {
        "class": AlzheimersDataset,
        "kwargs": {"subset": "HC"},
        "name": "Alzheimer's (HC)",
    },
    "sleepedf": {
        "class": SleepEDFDataset,
        "kwargs": {},
        "name": "SleepEDF",
    },
    "hmc": {
        "class": HMCDataset,
        "kwargs": {},
        "name": "HMC",
    },
    "homepap": {
        "class": HPAPDataset,
        "kwargs": {},
        "name": "HomePAP",
    },
}

# Dataset-specific channel configurations
# Only SleepEDF needs explicit channel filtering (excludes 1Hz channels)
DATASET_CHANNELS = {
    "sleepedf": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
    # All other datasets: None = load all available channels
}

TRAIN_CONFIG = {
    "pipeline_preset": "time_domain",
    "sequence_length": 1,  # epoch-to-epoch
    "max_epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "fold": 0,
    "early_stopping_patience": 10,
    "memmap_cache_size": 1000,
}

LOSS_W_MAIN = 1.0
LOSS_W_CHAN = 0.3


# ── Monkey-patch: custom _step for dict-output + modality ─────────────

_original_step = (
    Trainer._step.__func__
    if hasattr(Trainer._step, "__func__")
    else Trainer._step
)
_original_voting_eval_step = (
    Trainer._voting_eval_step.__func__
    if hasattr(Trainer._voting_eval_step, "__func__")
    else Trainer._voting_eval_step
)


def _compute_channel_loss(channel_logits, targets, data_mask, loss_fn):
    """Compute average CE across available (non-masked) channels.

    Args:
        channel_logits: (B, L, C, n_classes)
        targets: (B, L)
        data_mask: (B, C) bool — True = channel absent
        loss_fn: CrossEntropyLoss with ignore_index=-1

    Returns:
        Scalar loss averaged over available channels.
    """
    B, L, C, K = channel_logits.shape
    targets_flat = targets.reshape(-1)
    # Expand data_mask to (B, L, C)
    mask_expanded = data_mask.unsqueeze(1).expand(B, L, C)

    total_loss = 0.0
    count = 0
    for c in range(C):
        # Check if any sample has this channel
        available = ~mask_expanded[:, :, c]
        if not available.any():
            continue
        c_logits = channel_logits[:, :, c, :].reshape(-1, K)
        c_targets = targets_flat.clone()
        # Mask out unavailable epochs for this channel
        unavail = mask_expanded[:, :, c].reshape(-1)
        c_targets[unavail] = -1
        total_loss = total_loss + loss_fn(c_logits, c_targets)
        count += 1

    return total_loss / max(count, 1)


@staticmethod
def _sleeptokenizer_step(model, batch, loss_fn, device):
    """Custom _step for SleepTokenizer's dict output."""
    if isinstance(batch, dict) and "signals" in batch:
        from physioex.data.collate import stack_channels

        channels = batch.get("channel_order", [])
        for ch in channels:
            if ch not in batch["signals"]:
                raise ValueError(f"Channel '{ch}' in channel_order not found in signals keys: {list(batch['signals'].keys())}")

        inputs = stack_channels(batch).to(device)
        targets = batch["labels"].to(device)
        modality_ids = build_modality_ids(batch).to(device)
    else:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Fallback: assume all channels are EEG
        C = inputs.shape[2] if inputs.ndim == 4 else inputs.shape[1]
        modality_ids = torch.zeros(
            inputs.shape[0], C, dtype=torch.long, device=device
        )

    with torch.autocast(device.type if "cuda" in device.type else "cpu"):
        out = model(inputs, modality_ids=modality_ids)

    targets_flat = targets.reshape(-1)
    n_classes = out["logits"].shape[-1]

    # Check for valid targets
    valid_targets = targets_flat[targets_flat >= 0]
    if len(valid_targets) == 0:
        main_loss = torch.tensor(0.0, device=device)  # Skip loss if no valid targets
    else:
        # Check for NaN/Inf in embeddings or logits
        embeddings_flat = out["embedding"].reshape(-1, out["embedding"].shape[-1])
        if torch.isnan(embeddings_flat).any() or torch.isinf(embeddings_flat).any():
            print("WARNING: NaN/Inf in embeddings!")

        logits_flat = out["logits"].reshape(-1, n_classes)
        if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
            print("WARNING: NaN/Inf in logits!")
            print(f"  logits range: [{logits_flat.min():.4f}, {logits_flat.max():.4f}]")
            print(f"  centroids sample: {model.clf.centroids[0]}")

        # Main loss: prototypical distance-based logits
        main_loss = loss_fn(logits_flat, targets_flat)

    # Per-channel auxiliary loss
    chan_loss = _compute_channel_loss(
        out["channel_logits"], targets, out["data_mask"], loss_fn
    )

    loss = LOSS_W_MAIN * main_loss + LOSS_W_CHAN * chan_loss

    # Accuracy on main logits
    acc = accuracy_score(
        out["logits"].reshape(-1, n_classes),
        targets_flat,
        ignore_index=-1,
    )

    # Per-channel accuracy grouped by modality type
    from physioex.models.sleep_tokenizer import MODALITY_TYPES
    modality_names = {v: k for k, v in MODALITY_TYPES.items()}

    with torch.no_grad():
        # B, L, C, n_classes
        channel_logits = out["channel_logits"]
        data_mask = out["data_mask"]  # (B, C)
        B, L, C, _ = channel_logits.shape

        # Expand data_mask to (B, L, C)
        mask_expanded = data_mask.unsqueeze(1).expand(B, L, C)

        # Get predictions per channel
        chan_preds = channel_logits.argmax(dim=-1)  # (B, L, C)
        targets_expanded = targets.unsqueeze(-1).expand(B, L, C)  # (B, L, C)

        # Compute accuracy per channel - ONLY count valid targets (not -1)
        valid_mask = (targets_expanded != -1) & ~mask_expanded  # Valid targets AND available channels
        chan_correct = (chan_preds == targets_expanded) & valid_mask

        # Group by modality type
        # modality_ids is (B, C), need to expand to (B, L, C)
        modality_expanded = modality_ids.unsqueeze(1).expand(B, L, C)

        # Collect stats per modality
        modality_stats = {}  # {mod_id: (correct_count, total_count)}
        for mod_id in range(N_MODALITY_TYPES):
            mod_mask = (modality_expanded == mod_id) & valid_mask  # Only valid samples
            if not mod_mask.any():
                continue
            # Count correct predictions for this modality
            mod_correct = chan_correct[mod_mask].sum().item()
            mod_total = mod_mask.sum().item()  # Count only valid samples
            if mod_total > 0:
                modality_stats[mod_id] = (mod_correct, mod_total)

        # Compute average accuracy per modality
        modality_acc = {}
        for mod_id, (correct, total) in modality_stats.items():
            mod_name = modality_names.get(mod_id, f"MOD_{mod_id}")
            modality_acc[f"acc_{mod_name}"] = correct / total

        # Update prototypical centroids (EMA, no gradient)
        emb_flat = out["embedding"].reshape(-1, out["embedding"].shape[-1])
        model.clf.update_centroids(emb_flat, targets_flat)

    extra = {
        "loss_main": float(main_loss.detach()),
        "loss_chan": float(
            chan_loss.detach() if isinstance(chan_loss, torch.Tensor) else chan_loss
        ),
        **modality_acc,  # Add per-modality accuracies
    }

    return loss.cpu(), acc, extra


@torch.no_grad()
def _sleeptokenizer_voting_eval_step(model, batch, loss_fn, device, L=1):
    """Evaluation step with chunking to avoid OOM on long recordings."""
    eval_batch_size = 128
    channel_keys = [k for k in batch["signals"].keys()]
    night_length = batch["signals"][channel_keys[0]].shape[1]  # B, L, C, T

    loss, acc, extra = 0, 0, {}

    if night_length > eval_batch_size:
        valid_steps = 0
        for i in range(0, night_length, eval_batch_size):
            step = eval_batch_size if i + eval_batch_size <= night_length else night_length - i
            batch_chunk = copy.deepcopy(batch)

            batch_chunk["labels"] = batch_chunk["labels"][:, i:i+step].contiguous()

            # Skip chunks with all -1 labels to avoid NaN loss
            if (batch_chunk["labels"] == -1).all():
                continue

            for k in channel_keys:
                batch_chunk["signals"][k] = batch_chunk["signals"][k][:, i:i+step].contiguous()

            loss_chunk, acc_chunk, extra_chunk = _sleeptokenizer_step(model, batch_chunk, loss_fn, device)
            loss += loss_chunk
            acc += acc_chunk
            for k, v in extra_chunk.items():
                extra[k] = extra.get(k, 0) + v

            valid_steps += 1

        # Average the loss and accuracy over the chunks
        n_chunks = valid_steps if valid_steps > 0 else 1
        loss /= n_chunks
        acc /= n_chunks
        for k in extra:
            extra[k] /= n_chunks

        return loss, acc, extra

    # Short recording: process directly
    return _sleeptokenizer_step(model, batch, loss_fn, device)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=f"Pretrain {MODEL_NAME} on multi-dataset schema (see docs/training_schema.md)"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="pretrained_output/protosleepnet-gagliardi")
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT_DEFAULT,
        help=f"Path to sleep data root (default: $SLEEP_DATA_ROOT or {DATA_ROOT_DEFAULT})",
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--memmap_cache_size", type=int, default=None)
    parser.add_argument(
        "--exclude_datasets",
        nargs="+",
        default=None,
        help=f"Datasets to exclude from training (available: {', '.join(TRAINING_DATASETS.keys())})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # CLI overrides
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        TRAIN_CONFIG["batch_size"] = args.batch_size
    if args.lr is not None:
        TRAIN_CONFIG["lr"] = args.lr
    if args.weight_decay is not None:
        TRAIN_CONFIG["weight_decay"] = args.weight_decay
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience
    if args.memmap_cache_size is not None:
        TRAIN_CONFIG["memmap_cache_size"] = args.memmap_cache_size

    # ── Build MultiDataset from TRAINING_DATASETS ──
    print(f"Loading training datasets from: {args.data_root}")
    print(f"Memmap cache size: {TRAIN_CONFIG['memmap_cache_size']}")

    dataset_list = []
    excluded = set(args.exclude_datasets) if args.exclude_datasets else set()

    for ds_key, ds_config in TRAINING_DATASETS.items():
        if ds_key in excluded:
            print(f"  [EXCLUDED] {ds_config['name']}")
            continue

        try:
            DatasetClass = ds_config["class"]
            ds_kwargs = dict(ds_config["kwargs"])

            if ds_key == "alzheimers":
                dataset_path_name = "AlzheimerData"
            elif ds_key =="sleepedf":
                dataset_path_name = "physionet-sleep-data"
            elif ds_key == "parkinsons":
                dataset_path_name = "Parkinson_data"
            elif ds_key == "hmc":
                dataset_path_name = "hmc/physionet.org/files/hmc-sleep-staging/1.1/recordings"
            else:
                dataset_path_name = ds_key

            ds_kwargs["root"] = args.data_root + dataset_path_name
            
            ds_kwargs["pipelines"] = TRAIN_CONFIG["pipeline_preset"]
            ds_kwargs["sequence_length"] = TRAIN_CONFIG["sequence_length"]
            ds_kwargs["memmap_cache_size"] = TRAIN_CONFIG["memmap_cache_size"]

            # Apply dataset-specific channel filter if defined
            if ds_key in DATASET_CHANNELS:
                ds_kwargs["channels"] = DATASET_CHANNELS[ds_key]

            ds = DatasetClass(**ds_kwargs)
            n_subjects = ds.get_n_subjects()

            if n_subjects > 0:
                dataset_list.append(ds)
                channel_info = f"channels: {ds_kwargs.get('channels', 'all available')}"
                print(f"  [LOADED] {ds_config['name']}: {n_subjects} subjects ({channel_info})")
            else:
                print(f"  [SKIPPED] {ds_config['name']}: no subjects found")
        except Exception as e:
            print(f"  [ERROR] {ds_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    if not dataset_list:
        raise ValueError("No datasets loaded successfully! Check data_root paths.")

    dataset = MultiDataset(dataset_list)
    total_subjects = dataset.get_n_subjects()
    print(f"\nMultiDataset: {total_subjects} total subjects")
    print(f"  Available channels: {dataset.available_channels()}")

    # ── Model ──
    print("\nCreating model...")
    model = SleepTokenizer(**MODEL_KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SleepTokenizer: {n_params:,} parameters")

    # ── Optimizer ──
    print("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # ── Loss ──
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # ── Monkey-patch Trainer ──
    print("Patching Trainer._step and Trainer._voting_eval_step...")
    Trainer._step = _sleeptokenizer_step
    Trainer._voting_eval_step = _sleeptokenizer_voting_eval_step

    # ── Train ──
    print("\nStarting training...")
    print(f"  Dataset: {len(dataset_list)} datasets, {total_subjects} subjects")
    print(f"  Max epochs: {TRAIN_CONFIG['max_epochs']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  LR: {TRAIN_CONFIG['lr']}")
    print(f"  GPU: {args.gpu_id}")

    nw = args.num_workers
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=TRAIN_CONFIG["max_epochs"],
        optimizer=optimizer,
        loss=loss_fn,
        train_batch_size=TRAIN_CONFIG["batch_size"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
        valid_interval_ratio=0.1,
    )

    # ── Restore original _step ──
    Trainer._step = _original_step
    Trainer._voting_eval_step = _original_voting_eval_step

    # ── Save model ──
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    # ── Save config ──
    config = {
        "model_name": MODEL_NAME,
        "model_class": "physioex.models.sleep_tokenizer:SleepTokenizer",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "data_root": args.data_root,
        "datasets": {
            key: config["name"]
            for key, config in TRAINING_DATASETS.items()
            if key not in excluded
        },
        "excluded_datasets": list(excluded) if excluded else [],
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
