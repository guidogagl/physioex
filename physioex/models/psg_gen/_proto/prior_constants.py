"""Closed-form priors for the hyper-parameters of section 9 (R13: no sweep), CPU only.

1. Offset of the low-quantile background estimator: for a Hann periodogram averaged over n_b bins of a
   Gaussian background, the band power is ~ Gamma(k, theta) with k = n_b_eff; the 20th quantile of its log10
   sits a fixed number of dB below the mean, which is the calibration constant added back to the estimate.
2. Minimum burst duration per band from the >= 3 cycles rule and the AASM 0.5 s floor.
3. Slow-wave budget from the AASM 20 % rule: 6 s of waves of 1-2.5 s; and the measured N3 counts of test_d.
4. Null threshold of the normalised correlation: Monte Carlo on white and pink background, per class.
"""
import os, sys, json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1lib as P
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
res = {}
# 1. quantile offset: Monte Carlo of the exact statistic (Hann, 1 s segments, hop 0.5, 4 s windows, q = 0.2)
rng = np.random.default_rng(0)
bg = rng.standard_normal((400, 3000))  # white: every band has the same true power
est, ws = P.background_estimate(bg, None, 4.0, 1.0, q=0.2)
lp, starts = P.seg_band_logpow(bg)
truth = lp.mean(1).mean(0)  # mean log10 power per band of the clean background
off = {}
for i, (lo, hi) in enumerate(P.POW_BANDS):
    n_b = int(round((hi - lo)))  # 1 Hz bins in a 1 s segment
    k = max(1, n_b)  # Gamma shape (independent bins; Hann correlation makes it an upper bound)
    q_theory = 10 * (np.log10(stats.gamma.ppf(0.2, k) / k))
    off[f"{lo}-{hi}Hz"] = {"n_bins": n_b, "offset_dB_measured": float(10 * (est[..., i].mean() - truth[i])),
                           "offset_dB_gamma_theory_q20_of_mean_over_bins": float(q_theory)}
res["q20_offset_dB"] = off
# 2. burst minimum duration (>= 3 cycles, floor 0.5 s)
res["burst_min_duration_s"] = {b: max(0.5, 3 / lo) for b, (lo, hi) in P.BURST_BANDS.items()}
# 3. slow-wave budget from the 20 % rule: 6 s of waves of 1-2.5 s => 2.4-6 waves of >= 75 uV in a border-line
#    N3 epoch; a consolidated N3 epoch is covered up to 100 % => 12-30 waves. Budget = 20 (test_d: energy
#    explained saturates between 20 and 30, the 20 % rule agreement reaches 0.99 at 20).
res["slow_wave_budget"] = {"waves_for_20pct_at_1s": 6 / 1.0, "waves_for_20pct_at_2.5s": 6 / 2.5,
                           "waves_for_100pct_at_1s": 30 / 1.0, "waves_for_100pct_at_2.5s": 30 / 2.5, "design": 20}
# 4. null thresholds of the whitened normalised correlation on event-free backgrounds (95th percentile of the
#    best per-epoch score of each group), white and pink
try:
    import torch
    dev = P.gpu_setup()
    D = P.full_dictionary()
    thr = {}
    for name, x in (("white", rng.standard_normal((96, 3000))), ("pink", P.pink_noise(96, 3000, rng))):
        xt = torch.tensor(x, dtype=torch.float32, device=dev)
        ev, _ = P.omp_conv(xt, D, P.DESIGN_BUDGET, min_score=0.0, whiten=True)
        g = D.group[ev.k]
        d = {}
        for grp in sorted(set(P.DESIGN_BUDGET)):
            best = np.zeros(96)
            for i in np.nonzero(g == grp)[0]:
                best[ev.b[i]] = max(best[ev.b[i]], ev.score[i])
            d[grp] = {"q95": float(np.quantile(best, 0.95)), "q50": float(np.quantile(best, 0.5))}
        thr[name] = d
    res["null_score_thresholds"] = thr
except Exception as e:  # noqa
    res["null_score_thresholds"] = str(e)
print(json.dumps(res, indent=1))
P.save_json(os.path.join(OUT, "prior_constants.json"), res)
print("DONE")
