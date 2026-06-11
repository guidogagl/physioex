"""Joint prototype + spectral input analysis for disease discrimination.

For each K-Means prototype:
  1. Embedding-level: proportion, bout, intra-std discrimination (HOA vs PD)
  2. Input-level: spectral signature of exemplar epochs per class
     - Band powers (delta/theta/alpha/sigma/beta) per channel
     - Statistical comparison HOA vs PD within each prototype

Outputs a JSON report for offline discussion.

Usage:
    python analyze_proto_spectral.py \
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \
        --staging_dir .../proto-st-3ch-mixer/parkinsons_night/staging \
        --K 30 --output /out/report.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
CHANNELS = ["EEG", "EOG", "EMG"]

# Frequency bands — bin indices for fs=100, nfft=256 → freq_resolution ≈ 0.39 Hz
# freq[i] = i * fs / nfft = i * 100/256
# Bins: delta (0.5-4) → bins 1-10, theta (4-8) → 10-20, alpha (8-12) → 20-31,
#        sigma (12-16) → 31-41, beta (16-30) → 41-77
BANDS = {
    "delta": (1, 10),
    "theta": (10, 20),
    "alpha": (20, 31),
    "sigma": (31, 41),
    "beta": (41, 77),
}


def load_subjects(emb_dirs):
    subjects = {}
    for emb_dir in emb_dirs:
        for emb_path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(emb_path).replace("_embeddings.npy", "")
            if sid in subjects:
                continue
            base = emb_dir
            subjects[sid] = {
                "emb_path": emb_path,
                "inputs_path": os.path.join(base, f"{sid}_inputs.npy"),
                "metadata_path": os.path.join(base, f"{sid}_metadata.json"),
            }
    return subjects


def load_staging_predictions(staging_dir):
    preds = {}
    for fold_dir in sorted(glob.glob(os.path.join(staging_dir, "fold_*"))):
        path = os.path.join(fold_dir, "predictions.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for sid, p in data.items():
            if sid not in preds:
                preds[sid] = np.array(p["y_proba"], dtype=np.float32)
    return preds


def compute_band_powers(spectrogram):
    """Compute band powers from a spectrogram (T, F).

    The spectrogram is log-scale STFT magnitude.
    Convert back to linear power, average over time, then sum over frequency bands.
    """
    # spectrogram is log-scale → exponentiate
    linear = np.exp(spectrogram)
    # Average over time frames → (F,)
    mean_spectrum = linear.mean(axis=0)
    total = mean_spectrum.sum()

    powers = {}
    for band, (lo, hi) in BANDS.items():
        hi = min(hi, len(mean_spectrum))
        bp = mean_spectrum[lo:hi].sum()
        powers[f"bp_{band}"] = float(bp)
        powers[f"rp_{band}"] = float(bp / total) if total > 0 else 0.0

    powers["total_power"] = float(total)
    # Spectral centroid
    freqs = np.arange(len(mean_spectrum))
    if total > 0:
        powers["spectral_centroid"] = float(np.sum(freqs * mean_spectrum) / total)
    else:
        powers["spectral_centroid"] = 0.0

    return powers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--staging_dir", default=None)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_exemplar", type=int, default=100,
                        help="Exemplar epochs per prototype per class")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Proto-spectral analysis: K={args.K}, seed={args.seed}")

    # ── Load data ──
    subjects = load_subjects(args.emb_dirs)
    staging = load_staging_predictions(args.staging_dir) if args.staging_dir else {}

    all_embs, all_inputs, all_labels, all_sids = [], [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        if not os.path.exists(info["inputs_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        group = meta.get("group")
        if group is None:
            continue

        emb = np.load(info["emb_path"]).astype(np.float32)
        inp = np.load(info["inputs_path"]).astype(np.float32)

        # Align lengths
        n = min(len(emb), len(inp))
        emb, inp = emb[:n], inp[:n]

        all_embs.append(emb)
        all_inputs.append(inp)
        all_labels.append(str(group))
        all_sids.append(sid)

    classes = sorted(set(all_labels))
    print(f"  Subjects: {len(all_embs)}, classes: {classes}, "
          f"dist: {[all_labels.count(c) for c in classes]}")

    # ── K-Means ──
    print(f"  K-Means (K={args.K})...")
    concat = np.concatenate(all_embs, axis=0)
    km = MiniBatchKMeans(n_clusters=args.K, random_state=args.seed, batch_size=4096, n_init=3)
    km.fit(concat)
    centroids = km.cluster_centers_
    del concat

    # ── Build epoch registry: (subject_idx, epoch_idx, proto, dist, class) ──
    print("  Assigning epochs to prototypes...")
    epoch_registry = []
    profiles = []

    for i, emb in enumerate(all_embs):
        dists = cdist(emb, centroids)
        assignments = dists.argmin(axis=1)
        min_dists = dists[np.arange(len(emb)), assignments]

        N = len(emb)
        prop = np.zeros(args.K)
        intra_std = np.zeros(args.K)
        bout_mean = np.zeros(args.K)

        for k in range(args.K):
            mask = assignments == k
            prop[k] = mask.sum() / N
            if mask.sum() > 1:
                intra_std[k] = emb[mask].std(axis=0).mean()

        # Bouts
        bouts = {k: [] for k in range(args.K)}
        cur = assignments[0]
        blen = 1
        for j in range(1, N):
            if assignments[j] == cur:
                blen += 1
            else:
                bouts[cur].append(blen)
                cur = assignments[j]
                blen = 1
        bouts[cur].append(blen)
        for k in range(args.K):
            if bouts[k]:
                bout_mean[k] = np.mean(bouts[k])

        profiles.append({"proportion": prop, "bout_mean": bout_mean, "intra_std": intra_std})

        for j in range(N):
            epoch_registry.append((i, j, int(assignments[j]), float(min_dists[j])))

    # ── Per-prototype analysis ──
    print("  Analyzing prototypes...")
    idx_per_class = {c: [i for i, l in enumerate(all_labels) if l == c] for c in classes}

    # Sort epochs by distance per prototype
    proto_epochs = {k: [] for k in range(args.K)}
    for sid_idx, ep_idx, proto, dist in epoch_registry:
        proto_epochs[proto].append((dist, sid_idx, ep_idx))
    for k in range(args.K):
        proto_epochs[k].sort()

    results = []
    for k in range(args.K):
        info = {"proto": k}

        # ── Embedding discrimination ──
        best_d = 0
        for feat in ["proportion", "bout_mean", "intra_std"]:
            v0 = np.array([profiles[i][feat][k] for i in idx_per_class[classes[0]]])
            v1 = np.array([profiles[i][feat][k] for i in idx_per_class[classes[1]]])
            if v0.std() == 0 and v1.std() == 0:
                d, p = 0.0, 1.0
            else:
                _, p = stats.ttest_ind(v0, v1, equal_var=False)
                ps = np.sqrt((v0.std()**2 + v1.std()**2) / 2)
                d = (v0.mean() - v1.mean()) / ps if ps > 0 else 0.0
            info[f"{feat}_{classes[0]}"] = round(float(v0.mean()), 5)
            info[f"{feat}_{classes[1]}"] = round(float(v1.mean()), 5)
            info[f"{feat}_d"] = round(float(d), 3)
            info[f"{feat}_p"] = round(float(p), 4)
            if abs(d) > abs(best_d):
                best_d = d
        info["best_d"] = round(float(best_d), 3)
        info["best_abs_d"] = round(abs(best_d), 3)

        # ── Stage mapping ──
        if staging:
            stage_acc = np.zeros(5)
            cnt = 0
            for i, sid in enumerate(all_sids):
                if sid not in staging:
                    continue
                proba = staging[sid]
                emb = all_embs[i]
                n = min(len(emb), len(proba))
                asgn = cdist(emb[:n], centroids).argmin(axis=1)
                mask = asgn == k
                if mask.sum() > 0:
                    stage_acc += proba[:n][mask].sum(axis=0)
                    cnt += mask.sum()
            if cnt > 0:
                sdist = stage_acc / cnt
                info["stage_dist"] = {STAGE_NAMES[s]: round(float(sdist[s]), 3) for s in range(5)}
                info["dominant_stage"] = STAGE_NAMES[int(np.argmax(sdist))]
                info["n_epochs"] = int(cnt)

        # ── Input spectral analysis ──
        spectral = {}
        for c in classes:
            class_subjects = set(idx_per_class[c])
            exemplars = [(d, si, ei) for d, si, ei in proto_epochs[k] if si in class_subjects]
            exemplars = exemplars[:args.n_exemplar]

            if not exemplars:
                continue

            # Compute band powers per channel per exemplar epoch
            ch_powers = {ch_i: [] for ch_i in range(3)}
            for _, si, ei in exemplars:
                inp = all_inputs[si]
                if ei >= len(inp):
                    continue
                epoch_inp = inp[ei]  # (3, 29, 129)
                for ch_i in range(3):
                    bp = compute_band_powers(epoch_inp[ch_i])
                    ch_powers[ch_i].append(bp)

            # Average per channel
            for ch_i in range(3):
                if not ch_powers[ch_i]:
                    continue
                feat_names = list(ch_powers[ch_i][0].keys())
                spectral[f"{CHANNELS[ch_i]}_{c}"] = {
                    fn: round(float(np.mean([p[fn] for p in ch_powers[ch_i]])), 5)
                    for fn in feat_names
                }
                spectral[f"{CHANNELS[ch_i]}_{c}_n"] = len(ch_powers[ch_i])

        # Statistical test per channel per band
        spectral_tests = {}
        for ch_i in range(3):
            ch = CHANNELS[ch_i]
            key0, key1 = f"{ch}_{classes[0]}", f"{ch}_{classes[1]}"
            if key0 not in spectral or key1 not in spectral:
                continue

            # Need per-exemplar values for t-test
            class_subjects = {c: set(idx_per_class[c]) for c in classes}
            per_class_vals = {c: {ch_i: []} for c in classes}
            for c in classes:
                exs = [(d, si, ei) for d, si, ei in proto_epochs[k]
                       if si in class_subjects[c]][:args.n_exemplar]
                for _, si, ei in exs:
                    if ei >= len(all_inputs[si]):
                        continue
                    bp = compute_band_powers(all_inputs[si][ei][ch_i])
                    per_class_vals[c][ch_i].append(bp)

            tests = {}
            feat_names = ["rp_delta", "rp_theta", "rp_alpha", "rp_sigma", "rp_beta",
                          "spectral_centroid", "total_power"]
            for fn in feat_names:
                v0 = [p[fn] for p in per_class_vals[classes[0]][ch_i]]
                v1 = [p[fn] for p in per_class_vals[classes[1]][ch_i]]
                if len(v0) < 3 or len(v1) < 3:
                    continue
                v0, v1 = np.array(v0), np.array(v1)
                _, p = stats.ttest_ind(v0, v1, equal_var=False)
                ps = np.sqrt((v0.std()**2 + v1.std()**2) / 2)
                d = (v0.mean() - v1.mean()) / ps if ps > 0 else 0.0
                tests[fn] = {"d": round(float(d), 3), "p": round(float(p), 4)}
            spectral_tests[ch] = tests

        info["spectral"] = spectral
        info["spectral_tests"] = spectral_tests
        results.append(info)

    # Sort by discriminative power
    results.sort(key=lambda x: -x["best_abs_d"])

    # ── Print summary ──
    print(f"\n{'='*100}")
    print(f"  TOP DISCRIMINATIVE PROTOTYPES (embedding-level)")
    print(f"{'='*100}")
    for r in results[:15]:
        stage = r.get("dominant_stage", "?")
        n_ep = r.get("n_epochs", 0)
        disc = "*" if r["best_abs_d"] >= 0.3 else " "
        print(f"  {disc} proto {r['proto']:>2}  |d|={r['best_abs_d']:.3f}  "
              f"stage={stage:>3}  n_ep={n_ep:>5}  "
              f"prop: {classes[0]}={r.get(f'proportion_{classes[0]}',0):.3f} "
              f"{classes[1]}={r.get(f'proportion_{classes[1]}',0):.3f}")

    print(f"\n{'='*100}")
    print(f"  SPECTRAL DIFFERENCES IN DISCRIMINATIVE PROTOTYPES (|d|>=0.3)")
    print(f"{'='*100}")
    for r in results:
        if r["best_abs_d"] < 0.3:
            continue
        stage = r.get("dominant_stage", "?")
        print(f"\n  Proto {r['proto']} ({stage}, emb |d|={r['best_abs_d']:.3f}):")
        for ch in CHANNELS:
            tests = r.get("spectral_tests", {}).get(ch, {})
            if not tests:
                continue
            sig_bands = [(fn, t["d"], t["p"]) for fn, t in tests.items()
                         if abs(t["d"]) >= 0.3]
            if sig_bands:
                sig_bands.sort(key=lambda x: -abs(x[1]))
                parts = [f"{fn}(d={d:+.2f},p={p:.3f})" for fn, d, p in sig_bands]
                print(f"    {ch}: {', '.join(parts)}")

    # Save
    report = {
        "K": args.K, "seed": args.seed, "classes": classes,
        "n_subjects": len(all_embs),
        "class_dist": {c: all_labels.count(c) for c in classes},
        "prototypes": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
