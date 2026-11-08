"""End-to-end battery of the composite K-complex on a toy dump: an isolated slow transient in a low-SWA epoch
is a K-complex (planted expert labels), the same transient inside a slow-wave epoch is N3."""
import numpy as np

from physioex.train.bin import concept_compose as cc
from physioex.train.bin import concept_rationale as cr

T = cr.T


def _toy(seed=0, n_groups=4, n_nights=2, n_ep=40, n_dreams=3):
    rng = np.random.default_rng(seed)
    nights = n_groups * n_nights + n_dreams
    n_ep_all = nights * n_ep
    n = n_ep_all * T
    night = np.repeat(np.arange(nights), n_ep * T).astype(np.int32)
    gname = np.array([f"s{g // n_nights}" if g < n_groups * n_nights else "dreams" for g in night])
    sets = np.where(gname == "dreams", "dreams", "pool")
    kind = rng.integers(0, 3, n_ep_all)  # 0 = N2-like (low SWA), 1 = N3-like, 2 = W-like (alpha)
    delta = np.zeros(n, bool)
    alpha = np.zeros(n, bool)
    kc = np.zeros(n, bool)
    for e, k in enumerate(kind):
        a = e * T
        if k == 0:
            t = a + int(rng.integers(10, 50))
            delta[t:t + 2] = kc[t:t + 2] = True
        elif k == 1:
            delta[a + rng.choice(T, 30, replace=False)] = True
        else:
            alpha[a + rng.choice(T, 45, replace=False)] = True
    swa_ep = delta.reshape(-1, T).mean(1)
    alpha_ep = alpha.reshape(-1, T).mean(1)
    p0 = delta + 0.3 * rng.normal(size=n)
    P = np.column_stack([p0] + [rng.normal(size=n) for _ in range(3)]).astype(np.float32)
    F = np.column_stack([p0, p0] + [rng.normal(size=n) for _ in range(7)]).astype(np.float32)
    Eq = np.repeat(np.column_stack([swa_ep, rng.normal(size=n_ep_all) * 0.01]), T, 0).astype(np.float32)
    Nq = np.repeat(np.column_stack([alpha_ep, rng.normal(size=n_ep_all) * 0.01]), T, 0).astype(np.float32)
    ones = np.ones(n, bool)
    D = {"P": P, "F": F, "S": F[:, 2:].copy(), "y": np.repeat(np.where(kind == 0, 2, np.where(kind == 1, 3, 0)), T)
         .astype(np.int8), "night": night, "gname": gname, "set": sets, "logscale": np.zeros(n, np.float32),
         "Eq": Eq, "Nq": Nq,
         "refs": {"delta": (delta, ones), "alpha": (alpha, ones), "slow_wave": (delta & ~kc, ones),
                  "k_complex_rule": (kc, ones), "k_complex_vis1": (kc & (sets == "dreams"), ones),
                  "k_complex_vis2": (kc & (sets == "dreams"), ones)}}
    cr.finish(D)
    D["Eq"], D["Nq"] = Eq, Nq  # keep the planted latents readable (finish projects them on a PCA)
    return D


def test_battery_kc_planted():
    D = _toy()
    out = cc.battery_kc(D, n_null=20, n_null_pool=5)
    d = out["V2"]["dreams"]["k_complex_vis1"]
    assert d["lexicon"]["f1"] > 0.8 and d["oracle"]["f1"] > 0.95
    assert d["nulls"]["pred_perm_q95"] < d["lexicon"]["f1"] - 0.1
    assert d["nulls"]["word_roll_q95"] < d["lexicon"]["f1"] - 0.1
    v3 = out["V3"]["dreams:k_complex_vis1"]
    assert v3["no_swa"] > 0.05 and v3["word_only"] > 0.05
    assert out["V1"]["swa"]["spearman_pool_loso"] > 0.8
    assert out["V2"]["pool"]["kc_pure_vs_sw_pure"]["kc_pure"]["called_kc"] > 0.8
    assert cc._jsonable(out)  # serialisable
