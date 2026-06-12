"""REM PSD cluster analysis: low vs high RBDSQ.

Clusters REM epochs by spectral band powers (K-Means on 15-dim features),
then plots mean PSD per cluster comparing low RBDSQ (0-1) vs high (9-11).

Usage:
    python plot_rem_rbdsq.py \
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \
        --output_dir probing/figures
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# Freq axis: nfft=256, fs=100 → 129 bins, resolution 0.39 Hz/bin
FS, NFFT = 100, 256
FREQ = np.arange(129) * FS / NFFT

# Band definitions as bin ranges (inclusive)
BANDS = {
    "delta": (2, 10),   # 0.78 - 3.91 Hz
    "theta": (11, 20),  # 4.30 - 7.81 Hz
    "alpha": (21, 30),  # 8.20 - 11.72 Hz
    "sigma": (31, 40),  # 12.11 - 15.63 Hz
    "beta":  (41, 76),  # 16.02 - 29.69 Hz
}

# Channel order in _inputs.npy (alphabetical from PhysioEx)
CHANNELS = ["EEG", "EMG", "EOG"]


def compute_band_powers(spectrum):
    """Compute mean log-power in each band from a (129,) log-scale spectrum."""
    powers = []
    for lo, hi in BANDS.values():
        powers.append(spectrum[lo:hi+1].mean())
    return np.array(powers, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--K", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load subjects
    subjects = []
    for emb_dir in args.emb_dirs:
        for mp in sorted(glob.glob(os.path.join(emb_dir, "*_metadata.json"))):
            sid = os.path.basename(mp).replace("_metadata.json", "")
            with open(mp) as f:
                meta = json.load(f)
            rbdsq = meta.get("rbdsq_total")
            if rbdsq is None:
                continue

            labels_path = os.path.join(emb_dir, f"{sid}_labels.npy")
            inputs_path = os.path.join(emb_dir, f"{sid}_inputs.npy")
            if not all(os.path.exists(p) for p in [labels_path, inputs_path]):
                continue

            labels = np.load(labels_path).astype(np.int64)
            inputs = np.load(inputs_path).astype(np.float32)
            n = min(len(labels), len(inputs))

            subjects.append({
                "sid": sid, "group": meta.get("group", "?"),
                "rbdsq": float(rbdsq),
                "labels": labels[:n], "inputs": inputs[:n],
            })

    print(f"Loaded {len(subjects)} subjects")

    # Select extreme RBDSQ groups
    low_rbd = [s for s in subjects if s["rbdsq"] <= 1]
    high_rbd = [s for s in subjects if s["rbdsq"] >= 9]
    print(f"Low RBD (0-1):  {len(low_rbd)} subjects")
    print(f"High RBD (9-11): {len(high_rbd)} subjects")

    # Collect REM epochs — compute mean spectrum per epoch (3, 129)
    def get_rem_spectra(subj_list):
        spectra = []  # list of (3, 129)
        for s in subj_list:
            rem_mask = s["labels"] == 4
            if rem_mask.sum() == 0:
                continue
            rem_inputs = s["inputs"][rem_mask]  # (N_rem, 3, 29, 129)
            # Mean over time frames → (N_rem, 3, 129)
            epoch_spectra = rem_inputs.mean(axis=2)
            spectra.append(epoch_spectra)
        return np.concatenate(spectra) if spectra else np.empty((0, 3, 129))

    spectra_low = get_rem_spectra(low_rbd)
    spectra_high = get_rem_spectra(high_rbd)
    print(f"REM epochs: low={len(spectra_low)}, high={len(spectra_high)}")

    # Compute band power features for clustering (15 dim per epoch)
    def spectra_to_band_features(spectra):
        """(N, 3, 129) → (N, 15)"""
        N = len(spectra)
        feats = np.zeros((N, 3 * len(BANDS)), dtype=np.float32)
        for i in range(N):
            for ch in range(3):
                bp = compute_band_powers(spectra[i, ch])
                feats[i, ch * len(BANDS):(ch + 1) * len(BANDS)] = bp
        return feats

    feats_low = spectra_to_band_features(spectra_low)
    feats_high = spectra_to_band_features(spectra_high)

    # K-Means on all REM band power features
    all_feats = np.concatenate([feats_low, feats_high])
    km = MiniBatchKMeans(n_clusters=args.K, random_state=42, batch_size=2048, n_init=3)
    km.fit(all_feats)

    assign_low = km.predict(feats_low)
    assign_high = km.predict(feats_high)

    # Plot: mean PSD per cluster, low vs high
    fig, axes = plt.subplots(args.K, 3, figsize=(18, 4 * args.K))
    fig.suptitle("REM PSD Clusters (band power K-Means): Low RBDSQ (0-1, blue) vs High RBDSQ (9-11, red)",
                 fontsize=13, y=0.99)

    for proto in range(args.K):
        mask_low = assign_low == proto
        mask_high = assign_high == proto
        n_low, n_high = mask_low.sum(), mask_high.sum()

        for ch_i, ch_name in enumerate(CHANNELS):
            ax = axes[proto, ch_i]

            if n_low > 0:
                mean_psd = spectra_low[mask_low, ch_i, :].mean(axis=0)
                ax.plot(FREQ, mean_psd, 'b-', lw=1.5, alpha=0.8,
                        label=f"Low RBD (n={n_low})")

            if n_high > 0:
                mean_psd = spectra_high[mask_high, ch_i, :].mean(axis=0)
                ax.plot(FREQ, mean_psd, 'r-', lw=1.5, alpha=0.8,
                        label=f"High RBD (n={n_high})")

            ax.set_xlim(0, 35)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            if proto == 0:
                ax.set_title(ch_name, fontsize=12)
            if ch_i == 0:
                ax.set_ylabel(f"Cluster {proto}\nLog power", fontsize=10)
            if proto == args.K - 1:
                ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in [".png", ".pdf"]:
        out = os.path.join(args.output_dir, f"rem_psd_clusters_rbdsq{ext}")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")

    # Stats
    print(f"\nCluster distribution:")
    for p in range(args.K):
        nl = (assign_low == p).sum()
        nh = (assign_high == p).sum()
        print(f"  Cluster {p}: low={nl} ({nl/len(assign_low)*100:.1f}%)  "
              f"high={nh} ({nh/len(assign_high)*100:.1f}%)")

    # Print mean band powers per cluster for interpretation
    print(f"\nMean band powers per cluster (EEG only):")
    print(f"  {'Cluster':>7} {'delta':>8} {'theta':>8} {'alpha':>8} {'sigma':>8} {'beta':>8}")
    for p in range(args.K):
        mask = km.labels_ == p  # use training labels (all epochs)
        bp = all_feats[mask, :5].mean(axis=0)  # first 5 = EEG bands
        print(f"  {p:>7} {bp[0]:>8.2f} {bp[1]:>8.2f} {bp[2]:>8.2f} {bp[3]:>8.2f} {bp[4]:>8.2f}")


if __name__ == "__main__":
    main()
