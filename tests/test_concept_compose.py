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


def test_battery_kc_swa_from_words():
    D = _toy(seed=1)
    out = cc.battery_kc(D, n_null=10, n_null_pool=3, swa_source="words")
    assert out["V1"]["swa"]["source"] == "words" and out["V1"]["swa"]["spearman_pool"] > 0.8
    assert out["V2"]["dreams"]["k_complex_vis1"]["lexicon"]["f1"] > 0.8


def _toy_wake(seed=2):
    """W epochs without alpha but with blinks and high tone also hold slow transients (eye / movement delta):
    the alpha clause alone calls them K-complexes, the AASM wake rule on the EOG / EMG words does not."""
    D = _toy(seed=seed)
    rng = np.random.default_rng(seed)
    n = len(D["y"])
    w_ep = D["y"][::T] == 0
    kind_w = w_ep & (rng.random(len(w_ep)) < 0.5)  # eyes-open W; the other W epochs keep their alpha
    blink = np.zeros(n, bool)
    aton = np.zeros(n, bool)
    delta, h = D["refs"]["delta"]
    for e in np.flatnonzero(kind_w):
        a = e * T
        blink[a + rng.choice(T, 8, replace=False)] = True
        t = a + int(rng.integers(10, 50))
        delta[t:t + 2] = True  # spurious isolated slow transient in W
    D["refs"]["alpha"][0][np.repeat(kind_w, T)] = False  # eyes-open W: no alpha
    for e in np.flatnonzero(~w_ep):
        aton[e * T + rng.choice(T, 40, replace=False)] = True
    pe = np.column_stack([blink + 0.2 * rng.normal(size=n), rng.normal(size=n)]).astype(np.float32)
    pm = np.column_stack([aton + 0.2 * rng.normal(size=n), rng.normal(size=n)]).astype(np.float32)
    dre = D["set"] == "dreams"
    pe[dre], pm[dre] = np.nan, np.nan  # DREAMS: EEG only
    D["P@eog"], D["P@emg"] = pe, pm
    ones = np.ones(n, bool)
    D["refs"]["blink"] = (blink, ~dre)
    D["refs"]["rem_eye"] = (np.zeros(n, bool) | (blink & (rng.random(n) < 0.3)), ~dre)
    D["refs"]["atonia"] = (aton, ~dre)
    p0 = delta + 0.3 * rng.normal(size=n)
    D["P"][:, 0] = p0
    D["F"][:, 0] = D["F"][:, 1] = p0
    alpha_ep = D["refs"]["alpha"][0].reshape(-1, T).mean(1)
    Eq, Nq = D["Eq"], D["Nq"].copy()
    Nq[:, 0] = np.repeat(alpha_ep, T)
    cr.finish(D)  # re-derive context features and the ICA basis on the modified P
    D["Eq"], D["Nq"] = Eq, Nq
    return D


def test_battery_kc_wake_removes_w_transients():
    D = _toy_wake()
    base = cc.battery_kc(D, n_null=5, n_null_pool=3, swa_source="words")
    wake = cc.battery_kc(D, n_null=5, n_null_pool=3, swa_source="words", wake=True)
    pb = base["V2"]["pool"]["kc_vs_k_complex_rule"]
    pw = wake["V2"]["pool"]["kc_vs_k_complex_rule"]
    assert pb["stage_of_detections"]["W"] > 0.2 and pw["stage_of_detections"]["W"] < 0.05
    assert pw["lexicon"]["purity"] > pb["lexicon"]["purity"] + 0.2
    assert wake["V1"]["wake_epochs"]["dreams"]["lexicon"] < 0.5  # alpha fallback only


def test_battery_kc_n3_first_keeps_n3():
    D = _toy_wake(seed=3)
    out = cc.battery_kc(D, n_null=3, n_null_pool=2, swa_source="words", wake=True, n3_first=True)
    wc = out["V1"]["wake_clauses"]["lexicon"]
    assert set(wc) == {"alpha", "blink", "eye_tone", "swa_ge_20"} and wc["swa_ge_20"]["N3"] > 0.8
    n3 = out["V2"]["pool"]["n3_vs_slow_wave_in_swa_epochs"]["lexicon"]
    assert n3["coverage"] > 0.8


def test_battery_kc_eye_eeg_runs():
    D = _toy_wake(seed=4)
    out = cc.battery_kc(D, n_null=2, n_null_pool=2, swa_source="words", wake=True, n3_first=True, eye_eeg=True)
    assert out["V1"]["blink"]["ap_pool_loso"] > 0.5


def test_rem_w_rule_split():
    """Rapid eye movements in R (low chin tone) are called REM, the same movements in W (tone high) wake."""
    D = _toy_wake(seed=5)
    rng = np.random.default_rng(5)
    n = len(D["y"])
    ye = D["y"][::T].copy()
    ye[(ye == 2) & (rng.random(len(ye)) < 0.5)] = 4  # half of the N2-like epochs become REM (atonia already low)
    D["y"] = np.repeat(ye, T).astype(np.int8)
    eye = np.zeros(n, bool)
    for e in np.flatnonzero((ye == 4) | (ye == 0)):
        t = e * T + int(rng.integers(5, 50))
        eye[t:t + 3] = True
    dre = D["set"] == "dreams"
    pe = D["P@eog"].copy()
    pe[~dre, 1] = eye[~dre] + 0.2 * rng.normal(size=(~dre).sum())
    D["P@eog"] = pe
    D["refs"]["rem_eye"] = (eye, ~dre)
    out = cc.battery_rem(D, n_null_pool=3)
    sp = out["V2"]["pool"]["split_R_vs_W"]
    assert sp["lexicon"]["balanced"] > 0.8 and sp["no_tone"]["balanced"] < sp["lexicon"]["balanced"] - 0.2
    assert out["V2"]["pool"]["rem"]["lexicon"]["f1"] > out["V2"]["pool"]["rem"]["word_only"]["f1"] + 0.1
    assert cc._jsonable(out)


def test_alpha_in_context():
    """Alpha bursts: in eyes-closed W (alpha > 50 %) they are W alpha; a short burst in a low-tone REM epoch is
    REM alpha; in an N2 epoch with high tone it is an NREM intrusion."""
    D = _toy_wake(seed=6)
    rng = np.random.default_rng(6)
    n = len(D["y"])
    ye = D["y"][::T].copy()
    ye[(ye == 2) & (rng.random(len(ye)) < 0.5)] = 4
    D["y"] = np.repeat(ye, T).astype(np.int8)
    alpha = D["refs"]["alpha"][0].copy()
    aton = D["refs"]["atonia"][0].copy()
    pm = D["P@emg"].copy()
    for e in np.flatnonzero(ye == 2):  # N2: tone high (no atonia), short alpha burst
        aton[e * T:(e + 1) * T] = False
        t = e * T + int(rng.integers(5, 50))
        alpha[t:t + 3] = True
    for e in np.flatnonzero(ye == 4):
        t = e * T + int(rng.integers(5, 50))
        alpha[t:t + 3] = True
    dre = D["set"] == "dreams"
    pm[~dre, 0] = aton[~dre] + 0.2 * rng.normal(size=(~dre).sum())
    D["P@emg"] = pm
    D["P"][:, 1] = alpha + 0.2 * rng.normal(size=n)
    ones = np.ones(n, bool)
    y = D["y"]
    D["refs"]["alpha"] = (alpha, ones)
    D["refs"]["atonia"] = (aton, ~dre)
    D["refs"]["alpha_rem"] = (alpha & (y == 4), ones)
    D["refs"]["alpha_nrem"] = (alpha & ((y == 2) | (y == 3)), ones)
    out = cc.battery_alpha(D, n_null_pool=3)
    p = out["V2"]["pool"]
    assert set(p) == {"W", "REM", "NREM"}
    assert p["REM"]["lexicon"]["f1"] > p["REM"]["word_only"]["f1"] + 0.2
    assert p["NREM"]["lexicon"]["f1"] > p["NREM"]["nulls"]["state_perm_q95"]
    assert cc._jsonable(out)


def test_rem_tone_from_epoch_latent():
    """C2b: the chin tone read from the EMG epoch latent (absolute and relative to the night)."""
    D = _toy_wake(seed=7)
    rng = np.random.default_rng(7)
    n = len(D["y"])
    ye = D["y"][::T].copy()
    ye[(ye == 2) & (rng.random(len(ye)) < 0.5)] = 4
    D["y"] = np.repeat(ye, T).astype(np.int8)
    eye = np.zeros(n, bool)
    for e in np.flatnonzero((ye == 4) | (ye == 0)):
        t = e * T + int(rng.integers(5, 50))
        eye[t:t + 3] = True
    dre = D["set"] == "dreams"
    pe = D["P@eog"].copy()
    pe[~dre, 1] = eye[~dre] + 0.2 * rng.normal(size=(~dre).sum())
    D["P@eog"] = pe
    D["refs"]["rem_eye"] = (eye, ~dre)
    a_ep = D["refs"]["atonia"][0].reshape(-1, T).mean(1)
    q = np.column_stack([a_ep + 0.05 * rng.normal(size=len(a_ep)), rng.normal(size=len(a_ep))])
    q = np.repeat(q, T, 0).astype(np.float32)
    q[dre] = np.nan
    D["Eq@emg"] = q
    out = cc.battery_rem(D, n_null_pool=2, tone_source="eq")
    assert out["V1"]["tone"]["spearman_epoch_frac"] > 0.8
    assert out["V2"]["pool"]["split_R_vs_W"]["lexicon"]["balanced"] > 0.8


def test_cap_a1_from_band_reading():
    """A 3 s rise of slow power over its 60 s background in NREM is a CAP A1 phase; the lexicon reads the band
    powers from P."""
    D = _toy_wake(seed=8)
    rng = np.random.default_rng(8)
    n = len(D["y"])
    ye = D["y"][::T]
    burst = np.zeros(n, bool)
    for e in np.flatnonzero(ye == 2):
        if rng.random() < 0.5:
            t = e * T + int(rng.integers(5, 45))
            burst[t:t + 6] = True
    S = D["S"].copy()
    S[:, 0] = 0.1 * rng.normal(size=n) + 2.0 * burst
    S[:, 1:6] = 0.1 * rng.normal(size=(n, 5))
    D["S"] = S.astype(np.float32)
    D["P"][:, 2] = S[:, 0] + 0.05 * rng.normal(size=n)
    ones = np.ones(n, bool)
    aton = D["refs"]["atonia"][0].copy()
    aton[np.repeat(ye == 2, T)] = False  # NREM: chin tone not low (else the lexicon state calls it REM)
    dre = D["set"] == "dreams"
    pm = D["P@emg"].copy()
    pm[~dre, 0] = aton[~dre] + 0.2 * rng.normal(size=(~dre).sum())
    D["P@emg"] = pm
    D["refs"]["atonia"] = (aton, ~dre)
    D["refs"]["cap_a1"] = (burst, ones)
    D["refs"]["cap_a3"] = (np.zeros(n, bool), ones)
    out = cc.battery_cap(D, n_null_pool=3)
    a1 = out["V2"]["pool"]["a1"]
    assert a1["oracle"]["f1"] > 0.8 and a1["lexicon"]["f1"] > 0.6
    assert out["V1"]["bands_spearman_pool_loso"]["so"] > 0.8
    assert cc._jsonable(out)
