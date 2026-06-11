"""Prototype input-space analysis for disease discrimination.

Discovers K prototypes in embedding space, identifies discriminative ones,
then characterizes them in input space (raw EEG/EOG/EMG spectral features)
to understand what signal patterns they encode.

Outputs a detailed JSON report for offline analysis.

Usage:
    python analyze_prototypes.py \\
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \\
        --staging_dir .../proto-st-3ch-mixer/parkinsons_night/staging \\
        --dataset_name parkinsons --recording night \\
        --K 30 --output /out/analysis_report.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from scipy.signal import welch as welch_psd
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


# ── Subject loading (from embedding dirs) ───────────────────────────

def load_subjects(emb_dirs):
    subjects = {}
    for emb_dir in emb_dirs:
        for emb_path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(emb_path).replace("_embeddings.npy", "")
            if sid in subjects:
                continue
            subjects[sid] = {
                "emb_path": emb_path,
                "metadata_path": os.path.join(emb_dir, f"{sid}_metadata.json"),
            }
    return subjects


def load_staging_predictions(staging_dir):
    all_preds = {}
    for fold_dir in sorted(glob.glob(os.path.join(staging_dir, "fold_*"))):
        pred_path = os.path.join(fold_dir, "predictions.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        for sid, p in preds.items():
            if sid not in all_preds:
                all_preds[sid] = {
                    "y_proba": np.array(p["y_proba"], dtype=np.float32),
                }
    return all_preds


# ── Spectral features ───────────────────────────────────────────────

BAND_DEFS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

def compute_spectral_features(epoch, fs=100.0):
    """Compute spectral features for a single-channel epoch.

    Args:
        epoch: (L,) 1D signal array
        fs: sampling frequency

    Returns: dict of spectral features
    """
    # Welch PSD
    freqs, psd = welch_psd(epoch, fs=fs, nperseg=min(256, len(epoch)),
                            noverlap=min(128, len(epoch) // 2))

    total_power = np.trapz(psd, freqs)
    if total_power < 1e-12:
        return {
            **{f"bp_{b}": 0.0 for b in BAND_DEFS},
            **{f"rp_{b}": 0.0 for b in BAND_DEFS},
            "peak_freq": 0.0,
            "spectral_entropy": 0.0,
            "rms": 0.0,
        }

    features = {}

    # Band powers (absolute + relative)
    for band_name, (flo, fhi) in BAND_DEFS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        bp = np.trapz(psd[mask], freqs[mask]) if mask.sum() > 1 else 0.0
        features[f"bp_{band_name}"] = float(bp)
        features[f"rp_{band_name}"] = float(bp / total_power)

    # Peak frequency
    features["peak_freq"] = float(freqs[np.argmax(psd)])

    # Spectral entropy
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]
    features["spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm)))

    # RMS amplitude
    features["rms"] = float(np.sqrt(np.mean(epoch ** 2)))

    return features


def compute_epoch_spectral(signals_dict, channel_order, epoch_idx, fs=100.0):
    """Compute spectral features for one epoch across all channels.

    Returns: dict of {channel_name: {feature_name: value}}
    """
    result = {}
    for ch_name in channel_order:
        if ch_name not in signals_dict:
            continue
        sig = signals_dict[ch_name]
        if hasattr(sig, 'numpy'):
            sig = sig.numpy()
        epoch = sig[epoch_idx]
        result[ch_name] = compute_spectral_features(epoch, fs)
    return result


# ── Main analysis ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prototype input-space analysis for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--staging_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="parkinsons",
                        help="PhysioEx dataset name (parkinsons, alzheimers)")
    parser.add_argument("--recording", type=str, default="night",
                        choices=["night", "nap"])
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--n_exemplar", type=int, default=50,
                        help="Number of exemplar epochs per prototype for spectral analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON report path")
    args = parser.parse_args()

    STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

    print(f"Prototype analysis: K={args.K}, dataset={args.dataset_name}, "
          f"recording={args.recording}, n_exemplar={args.n_exemplar}")

    # ── Step 1: Load embeddings ──
    subjects = load_subjects(args.emb_dirs)
    print(f"Found {len(subjects)} subjects with embeddings")

    all_embs, all_labels, all_sids = [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        group = meta.get("group")
        if group is None:
            continue
        emb = np.load(info["emb_path"]).astype(np.float32)
        if len(emb) == 0:
            continue
        all_embs.append(emb)
        all_labels.append(str(group))
        all_sids.append(sid)

    print(f"  Loaded {len(all_embs)} subjects")
    classes = sorted(set(all_labels))
    print(f"  Classes: {classes}, dist: {[all_labels.count(c) for c in classes]}")

    # ── Step 2: Load staging predictions ──
    staging_preds = {}
    if args.staging_dir:
        staging_preds = load_staging_predictions(args.staging_dir)
        print(f"  Staging predictions: {len(staging_preds)} subjects")

    # ── Step 3: Load raw signals via PhysioEx ──
    print(f"  Loading PhysioEx {args.dataset_name} ({args.recording})...")
    if args.dataset_name == "parkinsons":
        from physioex.data.datasets import ParkinsonsDataset
        ds = ParkinsonsDataset(
            recording=args.recording,
            group=None,
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=0,
            cache_enabled=True,
        )
    elif args.dataset_name == "alzheimers":
        from physioex.data.datasets import AlzheimersDataset
        ds = AlzheimersDataset(
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=0,
            cache_enabled=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Build sid → dataset index mapping
    ds_subjects = ds.get_subjects()
    sid_to_ds_idx = {sid: i for i, sid in enumerate(ds_subjects)}
    print(f"  PhysioEx subjects: {len(ds_subjects)}")

    # Check overlap
    overlap = set(all_sids) & set(ds_subjects)
    print(f"  Overlap with embeddings: {len(overlap)} subjects")

    # ── Step 4: K-Means → prototypes ──
    print(f"  Fitting K-Means (K={args.K})...")
    all_concat = np.concatenate(all_embs, axis=0)
    km = MiniBatchKMeans(n_clusters=args.K, random_state=args.seed, batch_size=4096, n_init=3)
    km.fit(all_concat)
    centroids = km.cluster_centers_
    del all_concat
    print(f"  Prototypes: {centroids.shape}")

    # ── Step 5: Profile each subject ──
    print("  Computing subject profiles...")
    profiles = []  # one per subject
    # Also collect per-epoch info: (sid_idx, epoch_idx, proto_id, dist_to_proto)
    epoch_registry = []
    for i, emb in enumerate(all_embs):
        dists = cdist(emb, centroids)  # (N, K)
        assignments = dists.argmin(axis=1)
        min_dists = dists[np.arange(len(emb)), assignments]

        N = len(emb)
        proportion = np.zeros(args.K)
        for k in range(args.K):
            proportion[k] = (assignments == k).sum() / N

        # Bout stats
        bouts = {k: [] for k in range(args.K)}
        current = assignments[0]
        blen = 1
        for j in range(1, N):
            if assignments[j] == current:
                blen += 1
            else:
                bouts[current].append(blen)
                current = assignments[j]
                blen = 1
        bouts[current].append(blen)

        bout_mean = np.zeros(args.K)
        intra_std = np.zeros(args.K)
        for k in range(args.K):
            if bouts[k]:
                bout_mean[k] = np.mean(bouts[k])
            mask = assignments == k
            if mask.sum() > 1:
                intra_std[k] = emb[mask].std(axis=0).mean()

        profiles.append({
            "proportion": proportion,
            "bout_mean": bout_mean,
            "intra_std": intra_std,
        })

        # Register epochs for exemplar selection
        for j in range(N):
            epoch_registry.append((i, j, int(assignments[j]), float(min_dists[j])))

    # ── Step 6: Discrimination testing ──
    print("  Testing discrimination per prototype...")
    idx_0 = [i for i, l in enumerate(all_labels) if l == classes[0]]
    idx_1 = [i for i, l in enumerate(all_labels) if l == classes[1]]

    proto_analysis = []
    for k in range(args.K):
        info = {"index": k}

        best_abs_d = 0
        for feat_name in ["proportion", "bout_mean", "intra_std"]:
            vals_0 = np.array([profiles[i][feat_name][k] for i in idx_0])
            vals_1 = np.array([profiles[i][feat_name][k] for i in idx_1])

            if vals_0.std() == 0 and vals_1.std() == 0:
                d, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_ind(vals_0, vals_1, equal_var=False)
                pooled_std = np.sqrt((vals_0.std()**2 + vals_1.std()**2) / 2)
                d = (vals_0.mean() - vals_1.mean()) / pooled_std if pooled_std > 0 else 0.0

            info[f"{feat_name}_{classes[0]}_mean"] = float(vals_0.mean())
            info[f"{feat_name}_{classes[1]}_mean"] = float(vals_1.mean())
            info[f"{feat_name}_d"] = float(d)
            info[f"{feat_name}_p"] = float(p_val)

            if abs(d) > best_abs_d:
                best_abs_d = abs(d)

        info["best_abs_d"] = float(best_abs_d)
        info["discriminative"] = best_abs_d >= 0.3

        # Stage characterization
        if staging_preds:
            stage_accum = np.zeros(5, dtype=np.float64)
            stage_count = 0
            for i, sid in enumerate(all_sids):
                if sid not in staging_preds:
                    continue
                emb = all_embs[i]
                proba = staging_preds[sid]["y_proba"]
                n = min(len(emb), len(proba))
                assignments_i = cdist(emb[:n], centroids).argmin(axis=1)
                mask = assignments_i == k
                if mask.sum() > 0:
                    stage_accum += proba[:n][mask].sum(axis=0)
                    stage_count += mask.sum()
            if stage_count > 0:
                stage_dist = stage_accum / stage_count
                info["stage_distribution"] = {STAGE_NAMES[s]: float(stage_dist[s]) for s in range(5)}
                info["dominant_stage"] = STAGE_NAMES[int(np.argmax(stage_dist))]
                info["dominant_prob"] = float(stage_dist.max())
                info["n_epochs_total"] = int(stage_count)

        proto_analysis.append(info)

    # Sort by discriminative power
    proto_analysis.sort(key=lambda x: -x["best_abs_d"])
    n_disc = sum(1 for pa in proto_analysis if pa["discriminative"])
    print(f"  Discriminative prototypes (|d|>=0.3): {n_disc}/{args.K}")

    for pa in proto_analysis[:10]:
        stage = pa.get("dominant_stage", "?")
        prob = pa.get("dominant_prob", 0)
        disc = "DISC" if pa["discriminative"] else "    "
        print(f"    proto {pa['index']:>2}: {disc} |d|={pa['best_abs_d']:.3f}  "
              f"stage={stage} ({prob:.0%})")

    # ── Step 7: Input-space characterization ──
    print("  Analyzing exemplar epochs in input space...")

    # Group epoch_registry by prototype, sorted by distance
    proto_epochs = {k: [] for k in range(args.K)}
    for sid_idx, epoch_idx, proto_id, dist in epoch_registry:
        proto_epochs[proto_id].append((dist, sid_idx, epoch_idx))
    for k in range(args.K):
        proto_epochs[k].sort(key=lambda x: x[0])

    # For each prototype, load raw signals of exemplar epochs
    # Cache loaded subjects to avoid re-reading
    raw_cache = {}

    def load_raw_subject(sid):
        if sid in raw_cache:
            return raw_cache[sid]
        if sid not in sid_to_ds_idx:
            return None
        idx = sid_to_ds_idx[sid]
        try:
            batch = ds[idx]
            raw_cache[sid] = batch
            return batch
        except Exception as e:
            print(f"    Warning: could not load {sid}: {e}")
            return None

    for pa in proto_analysis:
        k = pa["index"]
        if not pa["discriminative"]:
            continue

        epochs_list = proto_epochs[k][:args.n_exemplar * 2]  # take more, filter later

        spectral_by_class = {c: {ch: [] for ch in ["EEG_0", "EOG_0", "EMG_0"]}
                             for c in classes}

        n_loaded = {c: 0 for c in classes}
        for dist, sid_idx, epoch_idx in epochs_list:
            sid = all_sids[sid_idx]
            label = all_labels[sid_idx]
            if n_loaded[label] >= args.n_exemplar:
                continue

            batch = load_raw_subject(sid)
            if batch is None:
                continue

            # Check epoch is in range
            for ch_name in ["EEG_0", "EOG_0", "EMG_0"]:
                if ch_name not in batch["signals"]:
                    continue
                sig = batch["signals"][ch_name]
                if hasattr(sig, 'numpy'):
                    sig = sig.numpy()
                if epoch_idx >= len(sig):
                    continue
                epoch_signal = sig[epoch_idx]
                sf = compute_spectral_features(epoch_signal, fs=100.0)
                spectral_by_class[label][ch_name].append(sf)

            n_loaded[label] += 1

        # Aggregate spectral features per class and channel
        pa["input_analysis"] = {}
        for ch_name in ["EEG_0", "EOG_0", "EMG_0"]:
            ch_analysis = {}
            for c in classes:
                feats_list = spectral_by_class[c][ch_name]
                if not feats_list:
                    continue
                # Average each feature
                feat_names = list(feats_list[0].keys())
                ch_analysis[c] = {
                    fn: float(np.mean([f[fn] for f in feats_list]))
                    for fn in feat_names
                }
                ch_analysis[f"{c}_n"] = len(feats_list)

            # Statistical test between classes
            if all(c in ch_analysis for c in classes):
                feat_names = list(spectral_by_class[classes[0]][ch_name][0].keys())
                ch_analysis["tests"] = {}
                for fn in feat_names:
                    vals_0 = [f[fn] for f in spectral_by_class[classes[0]][ch_name]]
                    vals_1 = [f[fn] for f in spectral_by_class[classes[1]][ch_name]]
                    if len(vals_0) < 3 or len(vals_1) < 3:
                        continue
                    vals_0, vals_1 = np.array(vals_0), np.array(vals_1)
                    t_stat, p_val = stats.ttest_ind(vals_0, vals_1, equal_var=False)
                    pooled_std = np.sqrt((vals_0.std()**2 + vals_1.std()**2) / 2)
                    d = (vals_0.mean() - vals_1.mean()) / pooled_std if pooled_std > 0 else 0.0
                    ch_analysis["tests"][fn] = {
                        "d": float(d),
                        "p": float(p_val),
                    }

            pa["input_analysis"][ch_name] = ch_analysis

    # Free memory
    raw_cache.clear()

    # ── Step 8: Build report ──
    report = {
        "dataset": args.dataset_name,
        "recording": args.recording,
        "K": args.K,
        "seed": args.seed,
        "n_subjects": len(all_embs),
        "classes": classes,
        "class_distribution": {c: all_labels.count(c) for c in classes},
        "n_discriminative": n_disc,
        "prototypes": proto_analysis,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
