"""T4 of plan rev. 2 of xmai-lab/sleep-lens#17 (#29): the V4 editor on FSQ codes, the decoded windows equal to the
decode of the whole night, the C judge on decoded windows equal to C on the whole night, and the T2 prototypes
recomputed by :func:`surprise_counterfactual.pool_prototypes` equal to those of :mod:`semantic_gates`."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from physioex.models.patch_codec import FSQ, PATCH
from physioex.train.bin import patch_codec as pc
from physioex.train.bin import semantic_gates as sg
from physioex.train.bin import surprise_counterfactual as scf
from physioex.train.bin import surprise_type as st

CPU = torch.device("cpu")


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_median_codes_is_the_lower_median_of_every_level():
    rng = np.random.default_rng(0)
    fsq = FSQ(5, 4, 9)
    c = rng.integers(0, 9 ** 4, (7, 5))
    v = fsq.lookup(torch.from_numpy(c)).numpy()
    for n, k in ((7, 3), (4, 1)):  # odd count: the median; even count: the lower one, still a level
        m = scf.median_codes(c[:n], 9, 4)
        assert m.shape == (5,) and (m >= 0).all() and (m < 9 ** 4).all()
        np.testing.assert_array_equal(fsq.lookup(torch.from_numpy(m)[None])[0].numpy(), np.sort(v[:n], 0)[k])
    np.testing.assert_array_equal(scf.median_codes(c[:1], 9, 4), c[0])


def _judge(C):
    cj = scf.CJudge.__new__(scf.CJudge)  # no prototypes from disk
    cj.codec, cj.fn, cj.levels, cj.dev = C, sg.feature_fns(C)["latent"], int(C.vq_levels), CPU
    rng = np.random.default_rng(1)
    cj.P = {k: v / np.linalg.norm(v) for k, v in
            (("kc", rng.standard_normal(C.fsq.dims)), ("background", rng.standard_normal(C.fsq.dims)))}
    return cj


def test_c_window_equals_c_on_the_whole_night_and_edits_stay_local():
    rev5 = _load_sibling("test_patch_codec_rev5")
    A, C = rev5._lapped(ctx_dec=2).eval(), rev5._lapped(zero_phase=True).eval()
    n_ep = 4
    P = n_ep * scf.EPP + 3  # a partial epoch at the end, as in the dumps
    x = rev5._ar_noise(1, P * PATCH)[0, : P * PATCH]
    codes = A.codes(x[None])[0].numpy().astype(np.int64)
    xhat = pc.decode_signal(A, torch.from_numpy(codes))
    cj = _judge(C)
    stage = np.repeat([2, 2, 3, 4], scf.T)
    cj.set_night(xhat, n_ep, stage)
    F0 = cj.F.copy()
    assert F0.shape == (n_ep * scf.T, C.fsq.dims) and np.isfinite(F0).all()
    lo_p, hi_p = scf.EPP, 2 * scf.EPP  # epoch 1, as a K-complex window
    lo_c, hi_c = lo_p - scf.C_MARGIN, hi_p + scf.C_MARGIN
    t0, t1 = (lo_p - 1) * scf.TPP, (hi_p + 1) * scf.TPP
    pa, pb = 20, 22
    new = (codes[pa:pb] + 7) % A.fsq.n_codes
    codes2 = codes.copy()
    codes2[pa:pb] = new
    xhat2 = pc.decode_signal(A, torch.from_numpy(codes2))
    U = scf.decode_window(A, codes, lo_c, hi_c, [[], [(pa, new)]], CPU)
    # the decoded windows equal the decode of the whole night, with and without the edit
    np.testing.assert_allclose(U[0], xhat[lo_c * PATCH:hi_c * PATCH].numpy(), atol=1e-5)
    np.testing.assert_allclose(U[1], xhat2[lo_c * PATCH:hi_c * PATCH].numpy(), atol=1e-5)
    Fw = cj.window(U, lo_c, t0, t1)
    assert Fw.shape == (2, t1 - t0, C.fsq.dims)
    np.testing.assert_allclose(Fw[0], F0[t0:t1], atol=1e-6)
    cj.set_night(xhat2, n_ep, stage)
    np.testing.assert_allclose(Fw[1], cj.F[t0:t1], atol=1e-6)
    # the edit reaches the features only within the decoder's receptive field and C's reach around the span
    R = A.receptive_patches + 3
    far = np.r_[0:(pa - R) * scf.TPP, (pb + R) * scf.TPP:n_ep * scf.T]
    np.testing.assert_allclose(cj.F[far], F0[far], atol=1e-6)
    # (the untrained C codes almost everything alike: the edit is checked on the decoded signal)
    span = slice((pa - lo_c) * PATCH, (pb - lo_c) * PATCH)
    assert np.abs(U[1][span] - U[0][span]).max() > 1e-3
    s = cj.score(Fw, "kc")
    assert s.shape == (2, t1 - t0) and np.all(np.abs(s) <= 2 + 1e-6)


def test_pool_prototypes_equal_those_of_semantic_gates(tmp_path, monkeypatch):
    sgt = _load_sibling("test_semantic_gates")
    rng = np.random.default_rng(2)
    T = st.T
    y_pool = np.repeat([1, 2, 3, 4], [10, 15, 10, 5])
    for unit in ("mesa", "hmc"):
        for i in range(2):
            ev = {"vertex": sgt._evs(rng, 0, 10 * T, 20), "k_complex_rule": sgt._evs(rng, 10 * T, 17 * T, 20),
                  "spindle_rule": sgt._evs(rng, 18 * T, 25 * T, 20), "slow_wave": sgt._evs(rng, 25 * T, 35 * T, 20),
                  "m1:arousal_m1": sgt._evs(rng, 35 * T, 40 * T, 12)}
            sgt._write(tmp_path, unit, f"n{i}", sgt._night(rng, y_pool, ev))
    for i in range(2):
        sgt._write(tmp_path, "dreams", f"excerpt{i}", sgt._night(rng, np.full(40, 2), {
            "k_complex_vis1": sgt._evs(rng, 0, 20 * T, 30), "spindle_vis1": sgt._evs(rng, 20 * T, 40 * T, 30)}))
        sgt._write(tmp_path, "mass_ss01", f"s{i}", sgt._night(rng, np.repeat([2, 4], 20),
                                                              {"arousal_ann": sgt._evs(rng, 0, 40 * T, 40)}))
    codecs = {"A": sgt._Fsq(), "B": sgt._Fsq(), "C": sgt._Fsq()}
    monkeypatch.setattr(pc, "load_codec", lambda run, dev: ({}, codecs[run]))
    out = tmp_path / "gates.json"
    sg.main(["--scan", str(tmp_path / "scan"), "--out", str(out), "--n-boot", "20"]
            + sum((["--model", f"{m}={m}:{tmp_path / 'dumps' / m}"] for m in codecs), []))
    r = json.load(open(out))["prototypes"]
    fsq = codecs["C"].fsq
    protos, info = scf.pool_prototypes(str(tmp_path / "scan"), str(tmp_path / "dumps" / "C"),
                                       lambda c: fsq.lookup(c).float().numpy())
    assert info["n_pool_nights"] == r["n_pool_nights"] == 4
    assert info["names"] == r["names"] and info["n_tokens"] == r["n_tokens"]
    np.testing.assert_allclose(info["cosine"], r["cosine"]["C_latent"], atol=1e-3)
    assert set(protos) == set(info["names"]) and all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in protos.values())
    one, _ = scf.pool_prototypes(str(tmp_path / "scan"), str(tmp_path / "dumps" / "C"),
                                 lambda c: fsq.lookup(c).float().numpy(), max_nights=1)
    assert set(one) <= set(protos)
