"""Semantic gates of the learned tokenizer (plan rev. 2 of xmai-lab/sleep-lens#17, T2): binned AP, entropy and
information, the patch features of FSQ and transform codecs, and an end-to-end run on synthetic dumps where the
semantic group of A carries the event identity, B carries it outside the semantic group and C not at all."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from physioex.models.patch_codec import FSQ, PATCH, PatchCodec, TransformCodec
from physioex.train.bin import semantic_gates as sg
from physioex.train.bin import surprise_type as st

GROUPS, GROUP, LEVELS = 4, 4, 9


def test_binned_ap_equals_sklearn_on_the_bins():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    ref = rng.random(5000) < 0.1
    s = rng.standard_normal(5000) + 1.5 * ref
    hp, hn = sg.night_hists([s[:2500], s[2500:]], [ref[:2500], ref[2500:]], n_bins=64)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, 65)[1:-1]))
    b = np.searchsorted(edges, s, side="right")
    assert sg.binned_ap(hp.sum(0), hn.sum(0)) == pytest.approx(average_precision_score(ref, b), abs=1e-12)
    hp4, hn4 = sg.night_hists([s[:2500], s[2500:]], [ref[:2500], ref[2500:]])  # N_BINS: close to the exact AP
    assert abs(sg.binned_ap(hp4.sum(0), hn4.sum(0)) - average_precision_score(ref, s)) < 0.005
    boot = sg.boot_aps({"x": (hp, hn), "y": (hp, hn)}, 20, 0)
    np.testing.assert_array_equal(boot["x"], boot["y"])  # the same night resamples for every key


def test_entropy_and_mutual_information():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, 20000)
    assert sg.entropy_bits(np.bincount(s)) == pytest.approx(2.0, abs=0.01)
    assert sg.mutual_information(s * 7 + 3, s) == pytest.approx(sg.entropy_bits(np.bincount(s)), abs=1e-9)
    x = rng.integers(0, 50, 20000)
    i, i_mm = sg.mutual_information(x, s), sg.mutual_information(x, s, True)
    assert 0 <= i < 0.02 and abs(i_mm) < i  # the plug-in bias, mostly removed by Miller-Madow


def test_sem_symbols_and_feature_shapes():
    torch.manual_seed(0)
    m = PatchCodec(width=8, dim=16, codes=32, levels=GROUPS, quantizer="fsq", fsq_levels=LEVELS, fsq_group=GROUP,
                   q_dropout=0.0, whiten="signal", dec_out="white", sem_dims=6, stage_weight=0.0).eval()
    c = m.codes(torch.randn(1, 7 * PATCH))[0]
    f = sg.feature_fns(m)
    assert list(f) == ["sem", "latent"]
    assert f["sem"](c).shape == (7, 6) and f["latent"](c).shape == (7, GROUPS * GROUP)
    sym = sg.sem_symbols(m, c)
    v = f["sem"](c).numpy()
    digits = np.round(v * (LEVELS - 1) / 2 + (LEVELS - 1) / 2).astype(np.int64)
    np.testing.assert_array_equal(sym, (digits * LEVELS ** np.arange(6)).sum(-1))
    tc = TransformCodec(groups=8)
    ct = tc.codes(torch.randn(1, 3 * PATCH))[0]
    assert list(sg.feature_fns(tc)) == ["klt"] and sg.feature_fns(tc)["klt"](ct).shape == (3, PATCH)
    stg = sg.patch_stages(np.array([5, 6, 8]), np.array([2, 3, 4]), 3, 90)  # epochs 3, 4 and 7 not scored
    assert stg[0] == -1 and stg[30] == 2 and stg[44] == 2 and stg[45] == 3 and stg[59] == 3
    assert stg[60] == -1 and stg[74] == -1 and stg[75] == 4 and stg[89] == 4


class _Fsq:
    quantizer = "fsq"

    def __init__(self, sem_dims=8):
        self.fsq, self.sem_dims = FSQ(GROUPS, GROUP, LEVELS), sem_dims


def _code(digits):
    return int(sum(d * LEVELS ** i for i, d in enumerate(digits)))


CLASS_CODE = {"kc": _code((8, 8, 0, 0)), "spindle": _code((0, 0, 8, 8)), "arousal": _code((8, 0, 8, 0)),
              "vertex": _code((0, 8, 0, 8)), "slow_wave": _code((8, 8, 8, 8))}
MASK_CLASS = {"k_complex_rule": "kc", "spindle_rule": "spindle", "m1:arousal_m1": "arousal", "vertex": "vertex",
              "slow_wave": "slow_wave", "k_complex_vis1": "kc", "spindle_vis1": "spindle", "arousal_ann": "arousal"}
CARRIER = {"A": 0, "B": 2, "C": None}  # FSQ group that carries the event identity (A: inside g_s = groups 0-1)


def _night(rng, y, events, n_ep=40):
    """Period of n_ep + 2 epochs (e0 = 3), dumped epochs 5 .. 5 + n_ep - 1 with stages ``y``; events =
    {mask: [(token, length)]}; the codes of the 3 models on the 4 grids."""
    e0, ts, n = 3, np.arange(5, 5 + n_ep), (n_ep + 2) * st.EP
    codes = {}
    for name, carrier in CARRIER.items():
        g = {}
        for o in st.OFFSETS:
            P = (n - o) // PATCH
            c = (rng.integers(3, 6, (P, GROUPS, GROUP)) * LEVELS ** np.arange(GROUP)).sum(-1)
            c[:, 1] = _code((4, 4, 4, 4))
            g[o] = c
        codes[name] = g
    masks = {}
    for key, evs in events.items():
        m = np.zeros(n_ep * st.T, bool)
        for a, ln in evs:
            m[a:a + ln] = True
            t = a + 2 * st.T
            for name, carrier in CARRIER.items():
                if carrier is None:
                    continue
                for o in st.OFFSETS:
                    p0, p1 = (t * st.TOK - o) // PATCH, ((t + ln) * st.TOK - 1 - o) // PATCH
                    codes[name][o][p0:p1 + 1, carrier] = CLASS_CODE[MASK_CLASS[key]]
        masks[key] = m
    return {"e0": e0, "ts": ts, "y": np.asarray(y), "codes": codes, "masks": masks}


def _packed(**masks):
    out = {}
    for k, v in masks.items():
        out[k], out[f"_len_{k}"] = np.packbits(np.asarray(v, bool)), np.array(len(v))
    return out


def _write(root, unit, sid, nd):
    for sub in ("scan", "scan/refs", "scan/m1", "dumps/A", "dumps/B", "dumps/C"):
        (root / sub / unit).mkdir(parents=True, exist_ok=True)
    for name in CARRIER:
        np.savez(root / "dumps" / name / unit / f"{sid}.npz", e0=nd["e0"],
                 **{f"c{o}": c.astype(np.int16) for o, c in nd["codes"][name].items()})
    np.savez(root / "scan" / unit / f"{sid}.npz", ts=nd["ts"], y=nd["y"])
    np.savez(root / "scan" / "refs" / unit / f"{sid}.npz",
             **_packed(**{k: v for k, v in nd["masks"].items() if not k.startswith("m1:")}))
    m1 = {k[3:]: v for k, v in nd["masks"].items() if k.startswith("m1:")}
    if m1:
        np.savez(root / "scan" / "m1" / unit / f"{sid}.npz", **_packed(**m1))


def _evs(rng, lo, hi, n):
    starts = np.sort(rng.choice(np.arange(lo, hi - 8, 12), n, replace=False))
    return [(int(a), int(rng.integers(2, 5))) for a in starts]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("sem")
    rng = np.random.default_rng(0)
    T = st.T
    y_pool = np.repeat([1, 2, 3, 4], [10, 15, 10, 5])  # N1, N2, N3, R epochs
    for unit in ("mesa", "hmc"):
        for i in range(4):
            ev = {"vertex": _evs(rng, 0, 10 * T, 20), "k_complex_rule": _evs(rng, 10 * T, 17 * T, 20),
                  "spindle_rule": _evs(rng, 18 * T, 25 * T, 20), "slow_wave": _evs(rng, 25 * T, 35 * T, 20),
                  "m1:arousal_m1": _evs(rng, 35 * T, 40 * T, 12)}
            _write(root, unit, f"n{i}", _night(rng, y_pool, ev))
    for i in range(3):
        _write(root, "dreams", f"excerpt{i}", _night(rng, np.full(40, 2), {
            "k_complex_vis1": _evs(rng, 0, 20 * T, 30), "spindle_vis1": _evs(rng, 20 * T, 40 * T, 30)}))
        _write(root, "mass_ss01", f"s{i}", _night(rng, np.repeat([2, 4], 20), {"arousal_ann": _evs(rng, 0, 40 * T, 40)}))
    return root


def test_semantic_gates_end_to_end(synthetic, monkeypatch):
    from physioex.train.bin import patch_codec

    codecs = {"A": _Fsq(), "B": _Fsq(), "C": _Fsq()}
    monkeypatch.setattr(patch_codec, "load_codec", lambda run, dev: ({}, codecs[run]))
    out = synthetic / "gates.json"
    sg.main(["--scan", str(synthetic / "scan"), "--out", str(out), "--n-boot", "100"]
            + sum((["--model", f"{m}={m}:{synthetic / 'dumps' / m}"] for m in codecs), []))
    r = json.load(open(out))
    assert r["prototypes"]["n_pool_nights"] == 8
    g1 = r["g_s1"]
    kc = g1["dreams_kc"]["a_minus_max_c"]["sem"]
    # the cosine of the registered score lifts the small, centred background vectors: A sem stays well below 1
    assert kc["delta"] > 0.1 and kc["ci"][0] > 0
    assert g1["gate"]["pass"] and set(g1["gate"]["sets"]) == {"dreams_kc", "dreams_spindle", "mass_ss01_arousal"}
    assert g1["dreams_kc"]["twin_best"] in ("C_sem", "C_latent")
    assert r["g_s2"]["dreams_kc"]["sem"]["delta"] > 0.1  # B's identity is outside g_s
    assert r["g_s2"]["dreams_kc"]["latent"]["delta"] < 0.2  # ... but inside its latent
    g3 = r["g_s3"]["models"]
    assert g3["A_sem"]["balanced_accuracy"] > 0.9 and g3["B_latent"]["balanced_accuracy"] > 0.9
    assert g3["C_sem"]["balanced_accuracy"] < 0.7 and g3["B_sem"]["balanced_accuracy"] < 0.7
    assert g3["A_sem"]["dreams_kc_typed_kc"]["share"] > 0.9
    g4 = r["g_s4"]
    assert g4["A"]["I_minus_shuffled_over_H"] > g4["C"]["I_minus_shuffled_over_H"] + 0.02
    assert g4["A"]["max_bits"] == pytest.approx(8 * np.log2(LEVELS))


def test_surprise_type_sem_feature(synthetic, monkeypatch):
    from physioex.train.bin import patch_codec

    codecs = {"A": _Fsq(), "C": _Fsq()}
    monkeypatch.setattr(patch_codec, "load_codec", lambda run, dev: ({}, codecs[run]))
    out = synthetic / "type_sem.json"
    st.main(["--codec", "A", "--twin-codec", "C", "--feature", "sem", "--codes", str(synthetic / "dumps" / "A"),
             "--twin", str(synthetic / "dumps" / "C"), "--scan", str(synthetic / "scan"), "--levels", "1",
             "--out", str(out)])
    r = json.load(open(out))
    o = r["sets"]["dreams_kc"]
    assert o["codes"]["own_class_share"] > 0.9 and o["twin"]["own_class_share"] < 0.6
