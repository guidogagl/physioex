"""Shared utilities for prototype reconstruction experiments.

Provides model/codebook loading, L2 distance computation, dataset
building, and embedding index construction for data_driven, model_driven,
and hybrid reconstruction scripts.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent dir so we can import build_protosleepnet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_protosleepnet import build_model  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────

BASE = Path(os.environ.get(
    "EXPERIMENT_DIR", "/home/dev/experiments/protosleepnet"
))
MODELS_DIR = BASE / "models"
EMB_DIR = BASE / "epoch-embeddings"
OUTPUT_BASE = BASE / "proto-reconstruction"

CONFIGS = {
    "seq": {
        "model_name": "protosleepnet-seq-3ch-mixer",
        "dataset": "mass",
        "seq_len": 20,
    },
    "st": {
        "model_name": "protosleepnet-st-3ch-mixer",
        "dataset": "shhs",
        "seq_len": 21,
    },
}


def get_paths(backbone, m=24):
    cfg = CONFIGS[backbone]
    model_dir = MODELS_DIR / cfg["model_name"]
    return {
        "checkpoint": model_dir / "model.pt",
        "codebook": model_dir / f"vq_kmeans/{m}/codebook.npy",
        "emb_dir": EMB_DIR / cfg["model_name"],
    }


# ── Model loading ────────────────────────────────────────────────────

def load_frozen_model(backbone, device, checkpoint_path=None):
    """Load ProtoSleepNet checkpoint with all params frozen."""
    if checkpoint_path is None:
        checkpoint_path = get_paths(backbone)["checkpoint"]
    model = build_model(backbone=backbone)
    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


def load_codebook(backbone, m=24, codebook_path=None):
    """Load VQ codebook (M, d_model) as float32 numpy array."""
    if codebook_path is None:
        codebook_path = get_paths(backbone, m)["codebook"]
    return np.load(codebook_path).astype(np.float32)


# ── L2 distance (must match VQ assignment) ───────────────────────────

def compute_l2_sq_distances_np(Z, codebook):
    """Squared L2 distances: (N, d) × (M, d) → (N, M). Numpy."""
    Z_sq = (Z ** 2).sum(axis=1, keepdims=True)
    C_sq = (codebook ** 2).sum(axis=1, keepdims=True).T
    return Z_sq + C_sq - 2 * (Z @ codebook.T)


# ── Embedding index ──────────────────────────────────────────────────

def load_train_embeddings(backbone):
    """Load per-subject epoch embeddings from the train split.

    Returns:
        subjects: list of (subject_id, Z, Y) tuples
            Z: (n_epochs, d_model) float32
            Y: (n_epochs,) int64
    """
    paths = get_paths(backbone)
    split_dir = paths["emb_dir"] / "train"
    emb_files = sorted(split_dir.glob("*_embeddings.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No *_embeddings.npy files in {split_dir}")

    subjects = []
    for ef in emb_files:
        subject_id = ef.stem.replace("_embeddings", "")
        lf = ef.parent / f"{subject_id}_labels.npy"
        Z = np.load(str(ef)).astype(np.float32)
        Y = np.load(str(lf)).astype(np.int64)
        subjects.append((subject_id, Z, Y))
    return subjects


def build_prototype_index(subjects, codebook, top_k=256):
    """Build per-prototype index of nearest assigned training epochs.

    For each prototype k:
      1. Assign every epoch to its nearest prototype (L2)
      2. Among epochs assigned to k, sort by distance ascending
      3. Keep top_k closest

    Args:
        subjects: list of (subject_id, Z, Y) from load_train_embeddings
        codebook: (M, d_model) numpy array
        top_k: max epochs to keep per prototype

    Returns:
        proto_index: dict[int] -> list of (subject_id, epoch_idx, distance, label)
        cluster_sizes: (M,) total epochs assigned to each prototype
    """
    M = codebook.shape[0]

    # Flatten all embeddings with index mapping
    all_Z, all_Y = [], []
    index_map = []  # flat_idx -> (subject_id, epoch_idx)
    for subject_id, Z, Y in subjects:
        n = len(Z)
        for j in range(n):
            index_map.append((subject_id, j))
        all_Z.append(Z)
        all_Y.append(Y)

    all_Z = np.concatenate(all_Z, axis=0)  # (N, d_model)
    all_Y = np.concatenate(all_Y, axis=0)  # (N,)
    N = len(all_Z)
    print(f"  Total train epochs: {N}, codebook M={M}, d={codebook.shape[1]}")

    # Compute distance matrix and assignments
    dist_matrix = compute_l2_sq_distances_np(all_Z, codebook)  # (N, M)
    assignments = dist_matrix.argmin(axis=1)  # (N,)

    # Build per-prototype index
    proto_index = {}
    cluster_sizes = np.zeros(M, dtype=int)

    for k in range(M):
        assigned_mask = assignments == k
        cluster_sizes[k] = int(assigned_mask.sum())

        if cluster_sizes[k] == 0:
            proto_index[k] = []
            continue

        indices_k = np.where(assigned_mask)[0]
        dists_k = dist_matrix[indices_k, k]

        # Sort by distance (squared L2), take top_k
        order = dists_k.argsort()[:top_k]
        selected = indices_k[order]

        entries = []
        for i, flat_idx in enumerate(selected):
            subject_id, epoch_idx = index_map[flat_idx]
            dist = float(np.sqrt(max(0, dists_k[order[i]])))
            label = int(all_Y[flat_idx])
            entries.append((subject_id, epoch_idx, dist, label))
        proto_index[k] = entries

    return proto_index, cluster_sizes


# ── Dataset building ─────────────────────────────────────────────────

def build_train_loader(backbone, channels=None, pipeline="seqsleepnet"):
    """Build train-split data loader matching embedding extraction config.

    Returns:
        dataset: the dataset object
        train_loader: DataLoader (batch_size=1, recording mode)
    """
    from physioex.data.datasets import get_dataset
    from physioex.train.trainer import Trainer

    if channels is None:
        channels = ["EEG", "EOG", "EMG"]

    cfg = CONFIGS[backbone]
    dataset_name = cfg["dataset"]

    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(
                cohort=c, channels=channels,
                pipelines=pipeline, sequence_length=0,
            )
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

    train_loader, _, _ = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=0,
    )
    return dataset, train_loader


def build_full_loader(dataset_name, channels=None, pipeline="seqsleepnet",
                      num_workers=0, **dataset_kwargs):
    """Build DataLoader for ALL subjects (no train/test split).

    Returns:
        dataset: the dataset object
        loader: DataLoader (batch_size=1, recording mode, all subjects)
    """
    from physioex.data.datasets import get_dataset
    from physioex.data.collate import dict_collate_fn
    from torch.utils.data import DataLoader

    if channels is None:
        channels = ["EEG", "EOG", "EMG"]

    DatasetClass = get_dataset(dataset_name)
    dataset = DatasetClass(
        channels=channels, pipelines=pipeline,
        sequence_length=0, **dataset_kwargs,
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=dict_collate_fn, num_workers=num_workers,
    )
    return dataset, loader


# ── I/O helpers ──────────────────────────────────────────────────────

def save_prototype_results(output_dir, k, **arrays):
    """Save arrays and metadata for prototype k."""
    proto_dir = Path(output_dir) / f"proto_{k:03d}"
    proto_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        if name == "metadata":
            with open(proto_dir / "metadata.json", "w") as f:
                json.dump(arr, f, indent=2)
        elif name == "loss_curve":
            np.save(proto_dir / "loss_curve.npy", np.array(arr, dtype=np.float32))
        else:
            np.save(proto_dir / f"{name}.npy", arr)


def save_summary(output_dir, summary):
    """Save global summary JSON."""
    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Argparse ─────────────────────────────────────────────────────────

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def add_common_args(parser):
    """Add shared CLI arguments."""
    parser.add_argument(
        "--backbone", type=str, required=True, choices=["seq", "st"],
        help="Model backbone: seq (SeqSleepNet) or st (SleepTransformer)",
    )
    parser.add_argument("--m", type=int, default=24, help="Codebook size M")
    parser.add_argument("--top_k", type=int, default=256, help="Samples per prototype")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument(
        "--codebook_path", type=str, default=None,
        help="Override codebook .npy path (for HPC)",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Override model checkpoint path (for HPC)",
    )
    return parser


def resolve_output_dir(args, method):
    """Resolve output directory, using default if not specified."""
    if args.output_dir:
        return Path(args.output_dir)
    cfg = CONFIGS[args.backbone]
    return OUTPUT_BASE / cfg["model_name"] / method


def get_device(args):
    if args.gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")
