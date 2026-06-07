"""Extract epoch embeddings h(x) from ProtoSleepNet via epoch_encode().

Handles the full pipeline: per-channel encoding + mixer + attn pooling.
Saves per-subject .npy files compatible with posthoc prototype scripts.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/extract_proto_embeddings.py \
        --backbone seq --checkpoint /path/to/model.pt \
        --output_dir /path/to/embeddings --dataset mass
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import stack_channels

sys.path.insert(0, os.path.dirname(__file__))
from build_protosleepnet import build_model


def load_model(backbone, checkpoint_path, device):
    model = build_model(backbone=backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def build_dataset_and_loaders(dataset_name, channels, pipeline, fold=0):
    from physioex.train.trainer import Trainer

    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines=pipeline, sequence_length=0)
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
        dataset = MultiDataset(cohorts)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        dataset = DatasetClass(channels=channels, pipelines=pipeline, sequence_length=0, **ds_kwargs)

    train_loader, valid_loader, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=fold,
    )

    return dataset, train_loader, valid_loader, test_loader


@torch.no_grad()
def extract_split(model, loader, split_dir, device, batch_size=256):
    os.makedirs(split_dir, exist_ok=True)
    n_extracted = 0
    n_skipped = 0

    for batch in tqdm(loader, desc=os.path.basename(split_dir)):
        subject_id = batch["subject"][0]["id"]
        emb_path = os.path.join(split_dir, f"{subject_id}_embeddings.npy")
        lbl_path = os.path.join(split_dir, f"{subject_id}_labels.npy")

        if os.path.exists(emb_path) and os.path.exists(lbl_path):
            n_skipped += 1
            continue

        inputs = stack_channels(batch)
        x = inputs.squeeze(0).to(device)  # (night_len, C, T, F)
        y = batch["labels"].squeeze(0).numpy()

        N = x.shape[0]
        embs = []
        for i in range(0, N, batch_size):
            chunk = x[i:i + batch_size].unsqueeze(0)  # (1, chunk_len, C, T, F)
            h = model.epoch_encode(chunk)  # (1, chunk_len, d_model)
            embs.append(h.squeeze(0).cpu().numpy())

        embs = np.concatenate(embs, axis=0).astype(np.float32)
        np.save(emb_path, embs)
        np.save(lbl_path, y.astype(np.int64))
        n_extracted += 1

    return n_extracted, n_skipped


def main():
    parser = argparse.ArgumentParser(description="Extract ProtoSleepNet epoch embeddings")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--channels", nargs="+", default=["EEG", "EOG", "EMG"])
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = load_model(args.backbone, args.checkpoint, device)
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    dataset, train_loader, valid_loader, test_loader = build_dataset_and_loaders(
        args.dataset, args.channels, "seqsleepnet", args.fold,
    )

    for split_name, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
        split_dir = os.path.join(args.output_dir, split_name)
        print(f"\nExtracting {split_name} ({len(loader)} subjects)")
        n_ext, n_skip = extract_split(model, loader, split_dir, device)
        print(f"  Extracted: {n_ext}, Skipped: {n_skip}")

    print(f"\nDone. Output at {args.output_dir}/")


if __name__ == "__main__":
    main()
