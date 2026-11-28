"""Out-of-band content of the raw input per dataset (xmai-lab/sleep-mechinterp#41, raw-encoder check).

The STFT encoders read only part of the spectrum: each 2-s frame is detrended and only the bins of the modality are
kept (EEG/EOG 0.39-32 Hz, EMG 10.2-44.9 Hz). The raw encoders (rawcnn, patch) read the whole 0-50 Hz signal. This
script measures, per dataset x modality on up to --nights nights (scored epochs, z = x * 10^(offset_db/20) as the raw
encoders see it), the share of the signal energy that the STFT encoders never see:
    dc      per-epoch mean (removed by the per-frame detrend)
    bands   share of the PSD (Welch, 2-s Hamming, as the STFT front-end) per band, from 0 to 50 Hz
    in_band share inside the modality band of the STFT encoders
    line40  / line50: PSD at 39.5-40.5 Hz (60-Hz mains aliased at fs 100) and 49-50 Hz, relative to 30-38 Hz

    python examples/pretrained/zoo/raw_spectrum_stats.py --root /sofia/projects/2026_101/cache/v1 --out spectrum.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from physioex.data.clean_cache import load_night

DATASETS = ["alzheimers_ad", "alzheimers_hc", "ccshs", "dcsm", "dreams", "hmc", "hpap", "lsec", "mass_ss01",
            "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa", "mros", "parkinsons_nap_hoa",
            "parkinsons_nap_pd", "parkinsons_night_hoa", "parkinsons_night_pd", "shhs_visit1", "shhs_visit2",
            "wsc_visit1", "wsc_visit2", "wsc_visit3", "wsc_visit4"]
EDGES = [0, 0.4, 4, 8, 12, 16, 32, 45, 50.01]
IN_BAND = {"EEG": (0.39, 32.0), "EOG": (0.39, 32.0), "EMG": (10.2, 44.9)}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nights", type=int, default=20)
ap.add_argument("--epochs", type=int, default=300, help="random scored epochs per night")
args = ap.parse_args()
root = Path(args.root)
rng = np.random.default_rng(0)
out = {}
for ds in DATASETS:
    subs = sorted(p.stem for p in (root / ds / "labels").glob("*.npy"))
    subs = [subs[i] for i in sorted(rng.choice(len(subs), min(args.nights, len(subs)), replace=False))]
    for m in ("EEG", "EOG", "EMG"):
        psd, dc, var, n = None, [], [], 0
        for s in subs:
            nt, _ = load_night(root, ds, s, m)
            if nt is None:
                continue
            x = np.asarray(nt.x, dtype=np.float64)[nt.y >= 0]
            x = x[rng.choice(len(x), min(len(x), args.epochs), replace=False)] * 10 ** (nt.offset_db / 20)
            f, p = welch(x, fs=100, window="hamming", nperseg=200, noverlap=100, nfft=256, detrend=False, axis=-1)
            psd = p.sum(0) if psd is None else psd + p.sum(0)
            dc.append((x.mean(-1) ** 2).mean())
            var.append((x ** 2).mean())
            n += len(x)
        if psd is None:
            continue
        psd /= n
        tot = psd.sum()
        lo, hi = IN_BAND[m]
        ref = psd[(f >= 30) & (f <= 38)].mean()
        out[f"{ds}|{m}"] = {
            "nights": len(dc), "dc_share": float(np.mean(dc) / np.mean(var)),
            "bands": {f"{a}-{b:g}": float(psd[(f >= a) & (f < b)].sum() / tot) for a, b in zip(EDGES[:-1], EDGES[1:])},
            "in_band": float(psd[(f >= lo) & (f <= hi)].sum() / tot),
            "line40": float(psd[(f >= 39.5) & (f <= 40.5)].mean() / ref),
            "line50": float(psd[f >= 49].mean() / ref)}
        r = out[f"{ds}|{m}"]
        print(f"{ds:22s} {m} dc {r['dc_share']:.3f} in_band {r['in_band']:.3f} line40 {r['line40']:.2f} "
              f"line50 {r['line50']:.2f} " + " ".join(f"{k}:{v:.3f}" for k, v in r["bands"].items()), flush=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print("DONE")
