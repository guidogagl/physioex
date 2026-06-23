"""Data-driven reconstruction for all M values from pre-extracted embeddings.

For a single seed, loads the training embeddings (already extracted),
runs K-Means for each M, builds prototype index, and saves the nearest
real epochs per prototype.

Usage:
    python seed_recon_all_m.py \
        --seed_dir /results/seed_123 --backbone seq --gpu_id 0
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto-reconstruction"))
sys.path.insert(0, os.path.dirname(__file__))

from build_protosleepnet import build_model
from train import build_dataset, BACKBONE_CONFIGS

from physioex.data.collate import stack_channels
from physioex.train.trainer import Trainer
from physioex.explain.prototypes.posthoc import learn_codebook_kmeans

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
M_VALUES = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]


def load_training_embeddings(emb_dir):
    """Load pre-extracted training embeddings."""
    import glob
    train_dir = os.path.join(emb_dir, "train")
    files = sorted(glob.glob(os.path.join(train_dir, "*_embeddings.npy")))
    if not files:
        raise FileNotFoundError(f"No embeddings in {train_dir}")

    subjects = []
    for ef in files:
        subj_id = os.path.basename(ef).replace("_embeddings.npy", "")
        lf = ef.replace("_embeddings.npy", "_labels.npy")
        Z = np.load(ef).astype(np.float32)
        Y = np.load(lf).astype(np.int64) if os.path.exists(lf) else np.zeros(len(Z), dtype=np.int64)
        subjects.append((subj_id, Z, Y))
    return subjects


def build_prototype_index(subjects, codebook, top_k=256):
    """Assign epochs to nearest prototype and keep top_k closest per prototype."""
    M = codebook.shape[0]
    proto_entries = {k: [] for k in range(M)}

    for subj_id, Z, Y in subjects:
        # L2 distances: (N, M)
        dists_sq = (
            np.sum(Z ** 2, axis=1, keepdims=True)
            + np.sum(codebook ** 2, axis=1, keepdims=True).T
            - 2.0 * Z @ codebook.T
        )
        assignments = dists_sq.argmin(axis=1)
        dists = np.sqrt(np.maximum(dists_sq[np.arange(len(Z)), assignments], 0))

        for i in range(len(Z)):
            k = int(assignments[i])
            proto_entries[k].append((subj_id, i, float(dists[i]), int(Y[i])))

    # Sort by distance and keep top_k
    cluster_sizes = np.array([len(proto_entries[k]) for k in range(M)])
    for k in range(M):
        proto_entries[k].sort(key=lambda x: x[2])
        proto_entries[k] = proto_entries[k][:top_k]

    return proto_entries, cluster_sizes


def main():
    parser = argparse.ArgumentParser(description="Data-driven reconstruction for all M")
    parser.add_argument("--seed_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="seq", choices=["seq", "st"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=256)
    args = parser.parse_args()

    seed_dir = args.seed_dir
    seed_name = os.path.basename(seed_dir)
    emb_dir = os.path.join(seed_dir, "embeddings")
    cfg = BACKBONE_CONFIGS[args.backbone]
    dataset_name = cfg["default_dataset"]
    seq_len = cfg["seq_len"]
    channels = cfg["channels_3ch"]

    print(f"=== {seed_name}: data-driven reconstruction for M={M_VALUES} ===")

    # ── Load embeddings (once) ──────────────────────────────────
    print("Loading training embeddings...")
    subjects = load_training_embeddings(emb_dir)
    Z_all = np.concatenate([Z for _, Z, _ in subjects], axis=0)
    print(f"  {len(subjects)} subjects, {len(Z_all)} total epochs")

    # ── Check which M values need work ──────────────────────────
    m_todo = []
    for M in M_VALUES:
        recon_dir = os.path.join(seed_dir, f"data_driven_m{M}")
        done_marker = os.path.join(recon_dir, "summary.json")
        if os.path.exists(done_marker):
            print(f"  SKIP M={M}: already done")
        else:
            m_todo.append(M)

    if not m_todo:
        print("All M values already computed. Done.")
        return

    # Also handle M=12 symlink (already computed in data_driven/)
    existing_m12 = os.path.join(seed_dir, "data_driven")
    if 12 not in m_todo and os.path.isdir(existing_m12):
        target = os.path.join(seed_dir, "data_driven_m12")
        if not os.path.exists(target):
            os.symlink(existing_m12, target)
            print(f"  Linked data_driven/ -> data_driven_m12/")

    print(f"  M values to compute: {m_todo}")

    # ── Build dataset for loading raw epochs ────────────────────
    print("Building dataset for raw epoch loading...")
    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        from physioex.data.datasets import get_dataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines="seqsleepnet", sequence_length=0)
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
        dataset = MultiDataset(cohorts)
    else:
        from physioex.data.datasets import get_dataset
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        dataset = DatasetClass(channels=channels, pipelines="seqsleepnet", sequence_length=0, **ds_kwargs)

    train_loader, _, _ = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=0,
    )
    print(f"  {len(train_loader)} training subjects in dataloader")

    # ── Process each M ──────────────────────────────────────────
    for M in m_todo:
        print(f"\n--- M={M} ---")
        recon_dir = os.path.join(seed_dir, f"data_driven_m{M}")
        os.makedirs(recon_dir, exist_ok=True)

        # K-Means
        codebook_path = os.path.join(seed_dir, f"codebook_m{M}.npy")
        if os.path.exists(codebook_path):
            codebook = np.load(codebook_path).astype(np.float32)
            print(f"  Loaded existing codebook: {codebook.shape}")
        else:
            print(f"  Running K-Means (M={M})...")
            codebook = learn_codebook_kmeans(Z_all, n_prototypes=M)
            np.save(codebook_path, codebook)
            print(f"  Saved codebook: {codebook.shape}")

        # Prototype index
        proto_index, cluster_sizes = build_prototype_index(subjects, codebook, top_k=args.top_k)
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}")

        # Determine needed epochs
        needed = defaultdict(set)
        for k in range(M):
            for subj_id, epoch_idx, dist, label in proto_index[k]:
                needed[subj_id].add(epoch_idx)

        # Load raw epochs
        epoch_store = {}
        for batch in train_loader:
            subj_id = batch["subject"][0]["id"]
            if subj_id not in needed:
                continue
            x = stack_channels(batch).squeeze(0)  # (n_epochs, C, T, F)
            for idx in needed[subj_id]:
                if idx < x.shape[0]:
                    epoch_store[(subj_id, idx)] = x[idx].cpu().numpy().astype(np.float32)

        # Save per-prototype
        summary = []
        for k in range(M):
            proto_dir = os.path.join(recon_dir, f"proto_{k:03d}")
            os.makedirs(proto_dir, exist_ok=True)

            epochs_list, labels_list, dists_list = [], [], []
            for subj_id, epoch_idx, dist, label in proto_index[k]:
                key = (subj_id, epoch_idx)
                if key in epoch_store:
                    epochs_list.append(epoch_store[key])
                    labels_list.append(label)
                    dists_list.append(dist)

            if epochs_list:
                np.save(os.path.join(proto_dir, "epochs.npy"), np.stack(epochs_list))
                np.save(os.path.join(proto_dir, "labels.npy"), np.array(labels_list, dtype=np.int64))
                np.save(os.path.join(proto_dir, "distances.npy"), np.array(dists_list, dtype=np.float32))

            valid_labels = np.array(labels_list)
            valid = valid_labels[valid_labels >= 0]
            dom = STAGE_NAMES[np.bincount(valid, minlength=5).argmax()] if len(valid) > 0 else "?"
            summary.append({"prototype": k, "n_samples": len(epochs_list),
                            "dominant_stage": dom, "cluster_size": int(cluster_sizes[k])})

        with open(os.path.join(recon_dir, "summary.json"), "w") as f:
            json.dump({"M": M, "seed": seed_name, "per_prototype": summary}, f, indent=2)

        print(f"  Saved {M} prototypes to {recon_dir}/")

    # Symlink M=12 if needed
    existing_m12 = os.path.join(seed_dir, "data_driven")
    target_m12 = os.path.join(seed_dir, "data_driven_m12")
    if os.path.isdir(existing_m12) and not os.path.exists(target_m12):
        os.symlink(existing_m12, target_m12)

    print(f"\nDone. All M values computed for {seed_name}.")


if __name__ == "__main__":
    main()
