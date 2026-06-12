"""Plot REM prototype spectral signatures: low vs high RBDSQ.

Clusters REM epochs (GT label=4) into 5 prototypes via K-Means on embeddings.
For each prototype, computes mean PSD from input spectrograms and compares
subjects with RBDSQ 0-1 (no RBD) vs RBDSQ 9-11 (severe RBD).

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--K", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load subjects with RBDSQ + embeddings + labels + inputs
    subjects = []
    for emb_dir in args.emb_dirs:
        for mp in sorted(glob.glob(os.path.join(emb_dir, "*_metadata.json"))):
            sid = os.path.basename(mp).replace("_metadata.json", "")
            with open(mp) as f:
                meta = json.load(f)
            rbdsq = meta.get("rbdsq_total")
            if rbdsq is None:
                continue

            emb_path = os.path.join(emb_dir, f"{sid}_embeddings.npy")
            labels_path = os.path.join(emb_dir, f"{sid}_labels.npy")
            inputs_path = os.path.join(emb_dir, f"{sid}_inputs.npy")

            if not all(os.path.exists(p) for p in [emb_path, labels_path, inputs_path]):
                continue

            emb = np.load(emb_path).astype(np.float32)
            labels = np.load(labels_path).astype(np.int64)
            inputs = np.load(inputs_path).astype(np.float32)

            n = min(len(emb), len(labels), len(inputs))
            subjects.append({
                "sid": sid, "group": meta.get("group", "?"),
                "rbdsq": float(rbdsq),
                "emb": emb[:n], "labels": labels[:n], "inputs": inputs[:n],
            })

    print(f"Loaded {len(subjects)} subjects")

    # Extreme groups
    low_rbd = [s for s in subjects if s["rbdsq"] <= 1]
    high_rbd = [s for s in subjects if s["rbdsq"] >= 9]

    print(f"Low RBD (0-1):  {len(low_rbd)} subjects, groups: {[s['group'] for s in low_rbd]}")
    print(f"High RBD (9-11): {len(high_rbd)} subjects, groups: {[s['group'] for s in high_rbd]}")

    # Collect REM epochs
    rem_embs_low, rem_inputs_low = [], []
    rem_embs_high, rem_inputs_high = [], []

    for s in low_rbd:
        mask = s["labels"] == 4
        if mask.sum() > 0:
            rem_embs_low.append(s["emb"][mask])
            rem_inputs_low.append(s["inputs"][mask])

    for s in high_rbd:
        mask = s["labels"] == 4
        if mask.sum() > 0:
            rem_embs_high.append(s["emb"][mask])
            rem_inputs_high.append(s["inputs"][mask])

    rem_embs_low = np.concatenate(rem_embs_low)
    rem_inputs_low = np.concatenate(rem_inputs_low)
    rem_embs_high = np.concatenate(rem_embs_high)
    rem_inputs_high = np.concatenate(rem_inputs_high)

    print(f"REM epochs: low={len(rem_embs_low)}, high={len(rem_embs_high)}")

    # K-Means on all REM epochs
    all_rem = np.concatenate([rem_embs_low, rem_embs_high])
    km = MiniBatchKMeans(n_clusters=args.K, random_state=42, batch_size=2048, n_init=3)
    km.fit(all_rem)

    assign_low = km.predict(rem_embs_low)
    assign_high = km.predict(rem_embs_high)

    # Freq axis
    fs, nfft = 100.0, 256
    freq_axis = np.arange(129) * fs / nfft

    channels = ["EEG", "EOG", "EMG"]

    # Plot
    fig, axes = plt.subplots(args.K, 3, figsize=(18, 4 * args.K))
    fig.suptitle("REM Prototype PSD: Low RBDSQ (0-1, blue) vs High RBDSQ (9-11, red)",
                 fontsize=14, y=0.99)

    for proto in range(args.K):
        mask_low = assign_low == proto
        mask_high = assign_high == proto
        n_low, n_high = mask_low.sum(), mask_high.sum()

        for ch_i, ch_name in enumerate(channels):
            ax = axes[proto, ch_i]

            if n_low > 0:
                psd = np.exp(rem_inputs_low[mask_low, ch_i]).mean(axis=(0, 1))
                ax.semilogy(freq_axis, psd, 'b-', lw=1.5, alpha=0.8,
                           label=f"Low RBD (n={n_low})")

            if n_high > 0:
                psd = np.exp(rem_inputs_high[mask_high, ch_i]).mean(axis=(0, 1))
                ax.semilogy(freq_axis, psd, 'r-', lw=1.5, alpha=0.8,
                           label=f"High RBD (n={n_high})")

            ax.set_xlim(0, 35)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            if proto == 0:
                ax.set_title(ch_name, fontsize=12)
            if ch_i == 0:
                ax.set_ylabel(f"Proto {proto}\nPower", fontsize=10)
            if proto == args.K - 1:
                ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in [".png", ".pdf"]:
        out = os.path.join(args.output_dir, f"rem_prototypes_rbdsq_psd{ext}")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")

    # Print stats
    print(f"\nPrototype distribution:")
    for p in range(args.K):
        nl = (assign_low == p).sum()
        nh = (assign_high == p).sum()
        print(f"  Proto {p}: low={nl} ({nl/len(assign_low)*100:.1f}%)  "
              f"high={nh} ({nh/len(assign_high)*100:.1f}%)")


if __name__ == "__main__":
    main()
