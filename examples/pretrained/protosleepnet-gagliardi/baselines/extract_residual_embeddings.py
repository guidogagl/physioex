"""Extract epoch embeddings from ResidualSequenceWrapper models.

Loads model via build_model() + checkpoint, extracts h(x) from
the epoch_encoder for each subject in train/valid/test splits.

Usage:
    python baselines/extract_residual_embeddings.py \
        --build_module train_seq_1ch_residual \
        --checkpoint /path/to/checkpoint.pt \
        --output_dir /path/to/embeddings \
        --dataset mass --channels EEG --seq_len 20
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import stack_channels

sys.path.insert(0, os.path.dirname(__file__))


def load_residual_model(build_module, checkpoint_path, device):
    """Load ResidualSequenceWrapper via build_model() + checkpoint."""
    mod = importlib.import_module(build_module)
    model = mod.build_model()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model.to(device).eval()


@torch.no_grad()
def extract_epoch_encoder(model, x):
    """Extract epoch embeddings h(x) from epoch_encoder.

    Args:
        x: (N, C, T, F)
    Returns:
        (N, d_model)
    """
    return model.epoch_encoder(x)


def extract_split(model, loader, split_dir, device, batch_size=256):
    """Extract and save per-subject epoch embeddings."""
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


def build_dataset_and_loaders(dataset_name, channels, pipeline, seq_len, fold=0):
    """Build dataset and return (dataset, train_loader, valid_loader, test_loader)."""
    from torch.utils.data import DataLoader, Subset
    from physioex.data.collate import dict_collate_fn

    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines=pipeline,
                      sequence_length=0)  # recording mode
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
        dataset = MultiDataset(cohorts)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        dataset = DatasetClass(
            channels=channels, pipelines=pipeline,
            sequence_length=0, **ds_kwargs,
        )

    train_ids, valid_ids, test_ids = dataset.split(fold=fold)

    def make_loader(indices):
        subset = Subset(dataset, indices.tolist() if hasattr(indices, 'tolist') else list(indices))
        return DataLoader(subset, batch_size=1, shuffle=False,
                          num_workers=0, collate_fn=dict_collate_fn)

    return dataset, make_loader(train_ids), make_loader(valid_ids), make_loader(test_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Extract epoch embeddings from ResidualSequenceWrapper"
    )
    parser.add_argument("--build_module", type=str, required=True,
                        help="Python module with build_model() (e.g. train_seq_1ch_residual)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--channels", nargs="+", default=["EEG"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load model
    model = load_residual_model(args.build_module, args.checkpoint, device)
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    dataset, train_loader, valid_loader, test_loader = build_dataset_and_loaders(
        args.dataset, args.channels, "seqsleepnet", 0, args.fold,
    )

    for split_name, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
        split_dir = os.path.join(args.output_dir, split_name)
        print(f"\n{'='*60}")
        print(f"Extracting {split_name} ({len(loader)} subjects)")
        print(f"{'='*60}")

        n_ext, n_skip, n_epochs = extract_split(
            model, loader, split_dir, device, args.batch_size,
        )
        print(f"  Extracted: {n_ext}, Skipped: {n_skip}, Total epochs: {n_epochs}")

    print(f"\nDone. Output at {args.output_dir}/")


if __name__ == "__main__":
    main()
