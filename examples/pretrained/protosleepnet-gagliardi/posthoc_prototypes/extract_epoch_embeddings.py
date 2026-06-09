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

    # Dataset with params
    python extract_epoch_embeddings.py --model_dir /path/to/st-baseline --output_dir /out \
        --dataset mass --cohort 1

    python extract_epoch_embeddings.py --model_dir /path/to/st-baseline --output_dir /out \
        --dataset parkinsons --recording night --group HOA
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

    if "factory" in config:
        factory = getattr(ModelClass, config["factory"])
        model = factory(**config.get("factory_kwargs", {}))
    else:
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

    # ProtoSleepNet: use epoch_encode() which handles per-channel + mixer + attn pool
    if hasattr(model, "epoch_encode"):
        h = model.epoch_encode(x.unsqueeze(0))  # (1, N, C, T, F) → (1, N, d)
        return h.squeeze(0)  # (N, d)

    # SleepTransformer / SeqSleepNet (any in_chan): feed all channels together
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

        try:
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
        except Exception as e:
            print(f"  [SKIP] {subject_id}: {e}")
            continue

    return n_extracted, n_skipped, n_total_epochs


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-sequence epoch embeddings from baseline models"
    )
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with config.json + model.pt")
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model name for load_from_pretrained (e.g. seqsleepnet-phan)")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HF repo ID (default: 4rooms/physioex)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Channel list (default: EEG EOG EMG)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name for output subdir (default: auto from dataset + params)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    # Dataset-specific parameters (flat, no JSON)
    parser.add_argument("--visit", type=int, default=None,
                        help="SHHS/WSC visit number")
    parser.add_argument("--cohort", type=int, default=None,
                        help="MASS cohort number (1-5)")
    parser.add_argument("--subset", type=str, default=None,
                        help="HPAP subset (lab-full, lab-split, home) or Alzheimers (AD, HC)")
    parser.add_argument("--recording", type=str, default=None,
                        help="Parkinsons recording type (night, nap, all)")
    parser.add_argument("--group", type=str, default=None,
                        help="Parkinsons group (HOA, PD)")
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Build dataset kwargs from flat args
    ds_kwargs = {}
    if args.visit is not None:
        ds_kwargs["visit"] = args.visit
    if args.cohort is not None:
        ds_kwargs["cohort"] = args.cohort
    if args.subset is not None:
        ds_kwargs["subset"] = args.subset
    if args.recording is not None:
        ds_kwargs["recording"] = args.recording
    if args.group is not None:
        ds_kwargs["group"] = args.group

    # Auto-generate dataset_name from dataset + params
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        parts = [args.dataset]
        if args.visit is not None:
            parts.append(f"visit{args.visit}")
        if args.cohort is not None:
            parts.append(f"cohort{args.cohort}")
        if args.subset is not None:
            parts.append(args.subset)
        if args.recording is not None:
            parts.append(args.recording)
        if args.group is not None:
            parts.append(args.group)
        dataset_name = "_".join(parts)

    # Detect mode: in-domain (train/valid/test) vs out-of-domain (all)
    is_indomain = args.dataset_name is None

    # Load model
    if args.model_name:
        from physioex.models import load_from_pretrained
        model_name = args.model_name
        kwargs = {"device": str(device)}
        if args.repo_id:
            kwargs["repo_id"] = args.repo_id
        model = load_from_pretrained(args.model_name, **kwargs)
        model.eval()
        print(f"Loading model: {model_name} from HF")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    elif args.model_dir:
        model_name = os.path.basename(args.model_dir)
        print(f"Loading model: {model_name} from {args.model_dir}")
        model, config = load_model(args.model_dir, device)
        print(f"  Class: {config['model_class']}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        parser.error("--model_dir or --model_name required")

    # Channels
    channels = args.channels if args.channels else CHANNELS

    # Load dataset in recording mode
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=channels,
        pipelines=PIPELINE,
        sequence_length=0,
        **ds_kwargs,
    )
    print(f"  Dataset: {dataset_name} ({len(dataset)} subjects)")

    model_out_dir = os.path.join(args.output_dir, model_name)

    if is_indomain:
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
