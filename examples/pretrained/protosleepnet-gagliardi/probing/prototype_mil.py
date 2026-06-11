"""Prototype MIL for disease discrimination.

Learns K prototypes in a projected embedding+spectral space to maximize
disease classification. Each prototype attends to relevant epochs via
distance-based soft attention, producing a subject representation.

Usage:
    python prototype_mil.py \
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \
        --source_name parkinsons_night --D 32 --K 5 --lr 1e-3 \
        --seed 42 --output_dir /out
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

BANDS = {
    "delta": (1, 10),
    "theta": (10, 20),
    "alpha": (20, 31),
    "sigma": (31, 41),
    "beta": (41, 77),
}


# ── Data loading ────────────────────────────────────────────────────

def _group_key(sid):
    if sid.startswith("SC"):
        return sid[:5]
    if sid.startswith("shhs") and "-" in sid:
        return sid.split("-", 1)[1]
    parts = sid.split("-")
    if len(parts) >= 3:
        return parts[-1]
    return sid


def compute_spectral_features(inputs):
    """Compute per-epoch spectral band powers from spectrogram.

    Args:
        inputs: (N, 3, 29, 129) log-scale spectrogram
    Returns:
        (N, 15) relative band powers — 5 bands × 3 channels
    """
    N, C, T, F = inputs.shape
    feats = np.zeros((N, C * len(BANDS)), dtype=np.float32)

    for ch in range(C):
        linear = np.exp(inputs[:, ch, :, :])  # (N, T, F)
        mean_spec = linear.mean(axis=1)  # (N, F)
        total = mean_spec.sum(axis=1, keepdims=True)  # (N, 1)
        total = np.maximum(total, 1e-12)

        for bi, (band, (lo, hi)) in enumerate(BANDS.items()):
            hi = min(hi, F)
            feats[:, ch * len(BANDS) + bi] = mean_spec[:, lo:hi].sum(axis=1) / total[:, 0]

    return feats


def load_all_subjects(emb_dirs):
    """Load embeddings + spectral features + labels for all subjects."""
    subjects = []

    for emb_dir in emb_dirs:
        for emb_path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(emb_path).replace("_embeddings.npy", "")
            meta_path = os.path.join(emb_dir, f"{sid}_metadata.json")
            inputs_path = os.path.join(emb_dir, f"{sid}_inputs.npy")
            spectral_path = os.path.join(emb_dir, f"{sid}_spectral.npy")

            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            group = meta.get("group")
            if group is None:
                continue

            emb = np.load(emb_path).astype(np.float32)

            # Load or compute spectral features
            if os.path.exists(spectral_path):
                spectral = np.load(spectral_path).astype(np.float32)
            elif os.path.exists(inputs_path):
                inp = np.load(inputs_path).astype(np.float32)
                n = min(len(emb), len(inp))
                spectral = compute_spectral_features(inp[:n])
                np.save(spectral_path, spectral)
            else:
                continue

            n = min(len(emb), len(spectral))
            features = np.concatenate([emb[:n], spectral[:n]], axis=1)  # (N, 143)

            subjects.append({
                "sid": sid,
                "features": features,
                "label": str(group),
                "group": _group_key(sid),
            })

    return subjects


# ── Model ───────────────────────────────────────────────────────────

class PrototypeMIL(nn.Module):
    def __init__(self, input_dim=143, proj_dim=32, n_prototypes=5):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, proj_dim))
        self.temperature = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(n_prototypes * proj_dim, 1)

    def init_prototypes(self, features_list, device):
        """Initialize prototypes via K-Means on projected train features."""
        with torch.no_grad():
            all_feat = np.concatenate(features_list, axis=0)
            # Project through current projection layer
            x = torch.from_numpy(all_feat).to(device)
            z = self.proj(x).cpu().numpy()
            K = self.prototypes.shape[0]
            km = MiniBatchKMeans(n_clusters=K, n_init=3, batch_size=4096)
            km.fit(z)
            self.prototypes.data = torch.from_numpy(
                km.cluster_centers_.astype(np.float32)
            ).to(device)

    def forward(self, x):
        """Forward pass for a single subject.

        Args:
            x: (N, input_dim) epoch features for one subject
        Returns:
            logit: scalar
        """
        z = self.proj(x)  # (N, D)

        # Distance to prototypes
        dists = torch.cdist(z.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)  # (N, K)

        # Attention: per prototype, softmax over epochs
        tau = self.temperature.clamp(min=0.1)
        attn = F.softmax(-dists / tau, dim=0)  # (N, K)

        # Weighted aggregation per prototype
        proto_repr = torch.mm(attn.T, z)  # (K, D)

        # Classify
        logit = self.classifier(proto_repr.flatten())  # (1,)
        return logit.squeeze(-1)


# ── Training ────────────────────────────────────────────────────────

def train_fold(model, train_data, val_data, device, lr, weight_decay,
               max_epochs, patience):
    """Train model for one fold with early stopping on val loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)

        for i in indices:
            x = torch.from_numpy(train_data[i]["features"]).to(device)
            y = torch.tensor([train_data[i]["y"]], dtype=torch.float32, device=device)
            logit = model(x)
            loss = criterion(logit.unsqueeze(0), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for d in val_data:
                x = torch.from_numpy(d["features"]).to(device)
                y = torch.tensor([d["y"]], dtype=torch.float32, device=device)
                logit = model(x)
                val_loss += criterion(logit.unsqueeze(0), y).item()

        val_loss /= max(len(val_data), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return epoch + 1


def evaluate(model, test_data, device):
    """Evaluate model on test subjects."""
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for d in test_data:
            x = torch.from_numpy(d["features"]).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            y_true.append(d["y"])
            y_proba.append(prob)
            y_pred.append(1 if prob > 0.5 else 0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1, "kappa": kappa}, y_true, y_pred, y_proba


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prototype MIL for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--D", type=int, default=32, help="Projection dim")
    parser.add_argument("--K", type=int, default=5, help="Number of prototypes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PrototypeMIL: D={args.D}, K={args.K}, lr={args.lr}, wd={args.weight_decay}, "
          f"seed={args.seed}, device={device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    subjects = load_all_subjects(args.emb_dirs)
    print(f"  Loaded {len(subjects)} subjects")

    classes = sorted(set(s["label"] for s in subjects))
    label_to_int = {l: i for i, l in enumerate(classes)}
    for s in subjects:
        s["y"] = label_to_int[s["label"]]

    y = np.array([s["y"] for s in subjects])
    groups = np.array([s["group"] for s in subjects])
    unique_groups = np.unique(groups)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups])
    sids = [s["sid"] for s in subjects]

    print(f"  Classes: {classes}, dist: {np.bincount(y).tolist()}")

    n_folds = min(args.n_folds, len(unique_groups))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.output_dir, args.source_name, "diagnosis")
    os.makedirs(output_dir, exist_ok=True)

    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(subjects)), y, group_ids)):

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split train_val into train/val (80/20)
        n_tv = len(train_val_idx)
        perm = np.random.RandomState(args.seed + fold_idx).permutation(n_tv)
        n_train = int(0.8 * n_tv)
        train_idx = train_val_idx[perm[:n_train]]
        val_idx = train_val_idx[perm[n_train:]]

        train_data = [subjects[i] for i in train_idx]
        val_data = [subjects[i] for i in val_idx]
        test_data = [subjects[i] for i in test_idx]

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": [sids[i] for i in train_idx],
            "val": [sids[i] for i in val_idx],
            "test": [sids[i] for i in test_idx],
        }

        # Create model
        input_dim = subjects[0]["features"].shape[1]
        model = PrototypeMIL(input_dim=input_dim, proj_dim=args.D,
                              n_prototypes=args.K).to(device)

        # Init prototypes via K-Means
        model.init_prototypes([s["features"] for s in train_data], device)

        # Train
        n_epochs = train_fold(model, train_data, val_data, device,
                               args.lr, args.weight_decay,
                               args.max_epochs, args.patience)

        # Evaluate
        metrics, y_true, y_pred, y_proba = evaluate(model, test_data, device)
        fold_metrics.append(metrics)

        print(f"    Fold {fold_idx}: acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1_macro']:.4f}  kappa={metrics['kappa']:.4f}  "
              f"(trained {n_epochs} epochs)")

        # Save
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
        predictions = {}
        for i, d in enumerate(test_data):
            predictions[d["sid"]] = {
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "y_proba": float(y_proba[i]),
            }
        with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    if not fold_metrics:
        print("  No valid folds")
        return

    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = len(subjects)
    summary["D"] = args.D
    summary["K"] = args.K
    summary["lr"] = args.lr
    summary["weight_decay"] = args.weight_decay
    summary["seed"] = args.seed
    summary["classes"] = classes
    summary["type"] = "prototype_mil"

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
