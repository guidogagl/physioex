"""Raw-input scale of the model zoo (xmai-lab/sleep-mechinterp#40), the raw-signal counterpart of the fixed dB window
of ChannelNet (physioex/models/seqsleepnet_clean.py).

The cache epochs are RobustScaled per night. The raw encoders read z = g * x, with g = 10^(offset_db / 20) the EMG
bandwidth gain (offset_db is the band-power correction of the STFT models, applied here as an amplitude gain). This
script measures |z| quantiles per dataset x modality on up to --nights nights (scored epochs only) and fixes:
    k     = median over datasets x modalities of q90(|z|)   (asinh is near-linear for |z| < k: typical amplitude)
    z_max = max over datasets x modalities of q99.99(|z|)   (largest kept amplitude)
    x_raw = clip(asinh(z / k) / asinh(z_max / k), -1, 1)
and reports the clipped fraction per dataset x modality (target < 1e-4 everywhere). No dataset statistics enter the
model: k and z_max are two constants written into physioex/models/zoo_blocks.py.

    python examples/pretrained/zoo/raw_scale_stats.py --root /sofia/projects/2026_101/cache/v1 --out raw_scale.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from physioex.data.clean_cache import load_night

DATASETS = ["alzheimers_ad", "alzheimers_hc", "ccshs", "dcsm", "dreams", "hmc", "hpap", "lsec", "mass_ss01",
            "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa", "mros", "parkinsons_nap_hoa",
            "parkinsons_nap_pd", "parkinsons_night_hoa", "parkinsons_night_pd", "shhs_visit1", "shhs_visit2",
            "wsc_visit1", "wsc_visit2", "wsc_visit3", "wsc_visit4"]
Q = [0.5, 0.9, 0.99, 0.999, 0.9999]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nights", type=int, default=20)
args = ap.parse_args()
root = Path(args.root)
rng = np.random.default_rng(0)
stats, samples = {}, {}
for ds in DATASETS:
    subs = sorted(p.stem for p in (root / ds / "labels").glob("*.npy"))
    subs = [subs[i] for i in sorted(rng.choice(len(subs), min(args.nights, len(subs)), replace=False))]
    for m in ("EEG", "EOG", "EMG"):
        vals = []
        for s in subs:
            nt, _ = load_night(root, ds, s, m)
            if nt is None:
                continue
            x = np.asarray(nt.x, dtype=np.float32)[nt.y >= 0]
            z = np.abs(x * 10 ** (nt.offset_db / 20)).ravel()
            vals.append(z[rng.choice(len(z), min(len(z), 200_000), replace=False)])
        if not vals:
            continue
        z = np.concatenate(vals)
        samples[f"{ds}|{m}"] = z
        stats[f"{ds}|{m}"] = {"nights": len(vals), **{f"q{q}": float(np.quantile(z, q)) for q in Q},
                              "max": float(z.max())}
        print(ds, m, {k: round(v, 3) for k, v in stats[f"{ds}|{m}"].items()}, flush=True)
k = float(np.median([v["q0.9"] for v in stats.values()]))
z_max = float(max(v["q0.9999"] for v in stats.values()))
clip = {key: float((z > z_max).mean()) for key, z in samples.items()}
out = {"k": k, "z_max": z_max, "A": float(np.arcsinh(z_max / k)), "clipped_fraction": clip,
       "max_clipped": max(clip.values()), "per_dataset": stats, "nights_per_dataset": args.nights}
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"k {k:.4f} z_max {z_max:.3f} A {out['A']:.3f} max clipped {out['max_clipped']:.2e}")
print("DONE")
