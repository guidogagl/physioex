"""Extract pre-sequence epoch embeddings h(x) from baseline models.

Supports two modes:
  1. SHHS in-domain (default): extracts train/valid/test splits separately
  2. Out-of-domain (--dataset_name): extracts ALL subjects into a single ``all/`` dir

Uses the PhysioEx dataset in recording mode (sequence_length=0)
with batch_size=1, so each batch is one full-night recording.

Output layout::

    In-domain (SHHS):
        {output_dir}/{model_name}/train/  valid/  test/

    Out-of-domain:
        {output_dir}/{model_name}/{dataset_name}/all/

Subjects already extracted are skipped automatically (resume-safe).

Usage:
    # SHHS in-domain (train/valid/test)
    python extract_epoch_embeddings.py --model_dir /path/to/st-baseline --output_dir /out

    # Out-of-domain dataset
    python extract_epoch_embeddings.py --model_dir /path/to/st-baseline --output_dir /out \
        --dataset hmc --dataset_name hmc

    # Dataset with kwargs
    python extract_epoch_embeddings.py --model_dir /path/to/st-baseline --output_dir /out \
        --dataset mass --dataset_kwargs '{"cohort": 1}' --dataset_name mass_cohort1
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import dict_collate_fn, stack_channels

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"


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

    Args:
        model: A model instance (any variant).
        x: (N, C, T, F) input spectrograms.

    Returns:
        (N, d_model) epoch embeddings (mean-pooled across channels).
    """
    N, C, T, F = x.shape

    if hasattr(model, "epoch_encoder") and hasattr(model, "in_chan"):
        x_flat = x.reshape(N * C, 1, T, F)
        embs = model.epoch_encoder(x_flat)
        embs = embs.reshape(N, C, -1)
        return embs.mean(dim=1)

    if hasattr(model, "epoch_encoder"):
        return model.epoch_encoder(x)

    if hasattr(model, "filterbank") and hasattr(model, "seqn1"):
        z = model.filterbank(x)
        z = z.permute(0, 2, 1, 3)
        N2, T2, C2, D = z.shape
        z = z.reshape(N2, T2, C2 * D)
        z, _ = model.seqn1(z)
        z = model.attention(z)
        return z

    raise ValueError(f"Unknown model type: {type(model).__name__}")


def extract_split(model, loader, split_dir, device, batch_size=256):
    """Extract and save per-subject epoch embeddings.

    Skips subjects whose embeddings already exist on disk.
    """
    os.makedirs(split_dir, exist_ok=True)

    n_extracted = 0
    n_skipped = 0
    n_total_epochs = 0

    for batch in tqdm(loader, desc=os.path.basename(split_dir)):
        subject_id = batch["subject"][0]["id"]

        emb_path = os.path.join(split_dir, f"{subject_id}_embeddings.npy")
        lbl_path = os.path.join(split_dir, f"{subject_id}_labels.npy")

        if os.path.exists(emb_path) and os.path.exists(lbl_path):
            n_skipped += 1
            existing = np.load(emb_path, mmap_mode="r")
            n_total_epochs += existing.shape[0]
            continue

        inputs = stack_channels(batch)
        x = inputs.squeeze(0).to(device)
        y = batch["labels"].squeeze(0).numpy()

        N = x.shape[0]
        embs = []
        for i in range(0, N, batch_size):
            chunk = x[i : i + batch_size]
            e = extract_epoch_encoder(model, chunk)
            embs.append(e.cpu().numpy())

        embs = np.concatenate(embs, axis=0).astype(np.float32)

        np.save(emb_path, embs)
        np.save(lbl_path, y.astype(np.int64))

        n_extracted += 1
        n_total_epochs += embs.shape[0]

    return n_extracted, n_skipped, n_total_epochs


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-sequence epoch embeddings from baseline models"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with config.json + model.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--dataset_kwargs", type=str, default="{}",
                        help="JSON string of dataset constructor kwargs")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name for output subdir (default: dataset arg)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    ds_kwargs = json.loads(args.dataset_kwargs)
    dataset_name = args.dataset_name or args.dataset

    # Detect mode: SHHS in-domain (train/valid/test) vs out-of-domain (all)
    is_shhs_indomain = (args.dataset == "shhs"
                        and ds_kwargs.get("visit", 1) == 1
                        and args.dataset_name is None)

    # Load model
    model_name = os.path.basename(args.model_dir)
    print(f"Loading model: {model_name} from {args.model_dir}")
    model, config = load_model(args.model_dir, device)
    print(f"  Class: {config['model_class']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset in recording mode
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=0,
        **ds_kwargs,
    )
    print(f"  Dataset: {dataset_name} ({len(dataset)} subjects)")

    model_out_dir = os.path.join(args.output_dir, model_name)

    if is_shhs_indomain:
        # SHHS in-domain: extract train/valid/test separately
        train_ids, valid_ids, test_ids = dataset.get_splits(fold=args.fold)
        all_subjects = dataset.get_subjects()
        id_to_idx = {sid: i for i, sid in enumerate(all_subjects)}

        print(f"  Splits: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}")

        for split_name, subject_ids in [
            ("train", train_ids),
            ("valid", valid_ids),
            ("test", test_ids),
        ]:
            split_dir = os.path.join(model_out_dir, split_name)
            print(f"\n{'='*60}")
            print(f"Extracting {split_name} ({len(subject_ids)} subjects)")
            print(f"{'='*60}")

            indices = [id_to_idx[sid] for sid in subject_ids if sid in id_to_idx]
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=0, collate_fn=dict_collate_fn,
            )

            n_ext, n_skip, n_epochs = extract_split(
                model, loader, split_dir, device, args.batch_size
            )
            print(f"  Extracted: {n_ext}, Skipped: {n_skip}, Total epochs: {n_epochs}")
    else:
        # Out-of-domain: extract ALL subjects into all/
        split_dir = os.path.join(model_out_dir, dataset_name, "all")
        print(f"\n{'='*60}")
        print(f"Extracting {dataset_name} (all {len(dataset)} subjects)")
        print(f"{'='*60}")

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=dict_collate_fn,
        )

        n_ext, n_skip, n_epochs = extract_split(
            model, loader, split_dir, device, args.batch_size
        )
        print(f"  Extracted: {n_ext}, Skipped: {n_skip}, Total epochs: {n_epochs}")

    print(f"\nDone. Output at {model_out_dir}/")


if __name__ == "__main__":
    main()
