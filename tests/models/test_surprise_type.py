"""Type of events by rule prototypes (xmai-lab/sleep-lens#22, S5): token features over the 4 patch grids, and
an end-to-end run on synthetic dumps where the codes carry the event identity and the twin's codes do not."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from physioex.train.bin import surprise_type as st

K, L, D = 16, 2, 8


def _packed(**masks):
    out = {}
    for k, v in masks.items():
        out[k], out[f"_len_{k}"] = np.packbits(np.asarray(v, bool)), np.array(len(v))
    return out


def test_token_features_average_the_covering_patches():
    # one dumped epoch at e0: tokens 0..59; grid 0 patch p covers tokens 4p..4p+3, grid 50 patch p tokens 4p+1..4p+4
    f0 = np.arange(15, dtype=float)[:, None]
    f50 = 100 + np.arange(14, dtype=float)[:, None]
    F = st.token_features({0: f0, 50: f50}, np.array([7]), 7)
    assert F.shape == (60, 1)
    assert F[0, 0] == 0.0  # grid 0 only
    assert F[1, 0] == pytest.approx((0 + 100) / 2) and F[4, 0] == pytest.approx((1 + 100) / 2)
    assert F[5, 0] == pytest.approx((1 + 101) / 2)


class _Codec:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.E = torch.randn(L, K, D, generator=g)

    def embed(self, c):
        return sum(self.E[l].to(c.device)[c[..., l]] for l in range(L))  # CPU or GPU, as surprise_type picks


def _night(rng, n_ep, events, code_of, unit_dir, sid, twin_random=True):
    """Period of n_ep + 2 epochs (e0 = 3), dumped epochs 5 .. 5 + n_ep - 1, stage N2; events = {mask: [(tok, len)]}."""
    e0, ts = 3, np.arange(5, 5 + n_ep)
    n = (n_ep + 2) * st.EP
    n_tok = n_ep * st.T
    codes = {o: rng.integers(0, 4, ((n - o) // st.PATCH, L)) for o in st.OFFSETS}
    masks, s = {}, rng.standard_normal((n_ep + 2) * st.T)
    for key, evs in events.items():
        m = np.zeros(n_tok, bool)
        for a, ln in evs:
            m[a:a + ln] = True
            g = a + 2 * st.T  # token index of the period
            s[g:g + ln] += 6.0
            for o in st.OFFSETS:
                p0, p1 = (g * st.TOK - o) // st.PATCH, ((g + ln) * st.TOK - 1 - o) // st.PATCH
                codes[o][p0:p1 + 1, 0] = code_of[key]
        masks[key] = m
    twin = {o: rng.integers(0, K, c.shape) for o, c in codes.items()} if twin_random else codes
    return {"e0": e0, "ts": ts, "y": np.full(n_ep, 2), "codes": codes, "twin": twin, "masks": masks, "s": s}


def _write(root, unit, sid, nd, m1_keys=()):
    for sub in ("codes", "twin", "scan", "scan/refs", "scan/m1", "sur", "tsur"):
        (root / sub / unit).mkdir(parents=True, exist_ok=True)
    np.savez(root / "codes" / unit / f"{sid}.npz", e0=nd["e0"], **{f"c{o}": c for o, c in nd["codes"].items()})
    np.savez(root / "twin" / unit / f"{sid}.npz", e0=nd["e0"], **{f"c{o}": c for o, c in nd["twin"].items()})
    np.savez(root / "scan" / unit / f"{sid}.npz", ts=nd["ts"], y=nd["y"])
    refs = {k: v for k, v in nd["masks"].items() if k not in m1_keys}
    m1 = {k[3:]: v for k, v in nd["masks"].items() if k in m1_keys}
    np.savez(root / "scan" / "refs" / unit / f"{sid}.npz", **_packed(**refs))
    if m1:
        np.savez(root / "scan" / "m1" / unit / f"{sid}.npz", **_packed(**m1))
    np.savez(root / "sur" / unit / f"{sid}.npz", s=nd["s"], e0=nd["e0"])
    np.savez(root / "tsur" / unit / f"{sid}.npz", s=np.random.default_rng(1).standard_normal(len(nd["s"])), e0=nd["e0"])


def test_type_end_to_end(tmp_path, monkeypatch):
    from physioex.models.patch_codec import SpectralTwin
    from physioex.train.bin import patch_codec

    rng = np.random.default_rng(0)
    code_of = {"k_complex_rule": 7, "spindle_rule": 11, "m1:kc_m1": 7, "k_complex_vis1": 7, "spindle_vis1": 11}

    def evs(n):
        st0 = np.sort(rng.choice(np.arange(10, 40 * st.T - 20, 40), n, replace=False))
        return [(int(a), int(rng.integers(2, 5))) for a in st0]

    for i in range(3):
        for unit in ("mesa", "hmc"):
            nd = _night(rng, 40, {"k_complex_rule": evs(15), "spindle_rule": evs(15)}, code_of, None, None)
            _write(tmp_path, unit, f"n{i}", nd)
        kc = _night(rng, 40, {"k_complex_vis1": evs(20)}, code_of, None, None)
        _write(tmp_path, "dreams", f"kc_excerpt{i}", kc)
        sp = _night(rng, 40, {"spindle_vis1": evs(20)}, code_of, None, None)
        _write(tmp_path, "dreams", f"excerpt{i}", sp)
    tw = SpectralTwin.fit(torch.randn(500, D), n_codes=K, levels=L, iters=3)
    torch.save(tw.state_dict(), tmp_path / "twin" / "twin.pt")
    monkeypatch.setattr(patch_codec, "load_codec", lambda run, dev: ({}, _Codec()))
    out = tmp_path / "type.json"
    st.main(["--codec", "x", "--codes", str(tmp_path / "codes"), "--twin", str(tmp_path / "twin"),
             "--scan", str(tmp_path / "scan"), "--surprise", str(tmp_path / "sur"), "--twin-surprise",
             str(tmp_path / "tsur"), "--levels", str(L), "--out", str(out)])
    r = json.load(open(out))
    assert r["prototypes"]["n_pool_nights"] == 6 and r["prototypes"]["n_tokens"]["kc"] > 0
    kv = r["dreams_kc_vs_spindle"]
    assert kv["codes"]["balanced_accuracy"] > 0.9 and kv["delta"]["ci"][0] > 0.2
    o = r["sets"]["dreams_kc"]
    assert o["codes"]["own_class_share"] > 0.9 > o["codes"]["chance_share"] + 0.5
    se = o["codes"]["surprise_events"]
    assert se["matched"] > 0 and se["own_class_share_matched"] > 0.8
