"""Structured generative PSG model (psg_gen): the ten tests of SPEC.md section 3.

Synthetic data only (tests/models/psg_gen_synth.py), small batches, CPU. Each test
``pytest.importorskip``s the modules it exercises so a not-yet-ported module skips the test
instead of failing collection. Events are matched against the planted ground truth with a
+-0.3 s onset tolerance and the right class/band, as the spec requires.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Synthetic generators (sibling file, 30 s @ 100 Hz epochs, floor RMS 1)
# ---------------------------------------------------------------------------


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


synth = _load_sibling("psg_gen_synth")

T = 3000
CLS_BURST, CLS_TRANSIENT, CLS_SLOWWAVE = 0, 1, 2
BUDGET = {CLS_BURST: 14, CLS_TRANSIENT: 4, CLS_SLOWWAVE: 20}  # priors, SPEC section 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dict_and_thresholds(B_bg: int = 16, seed: int = 100):
    dicts = pytest.importorskip("physioex.models.psg_gen.dicts")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    d = dicts.full_dictionary()
    thr = inference.null_thresholds(d, synth.epochs("pink", B_bg, seed=seed))
    return d, thr


def _band_power(sig: torch.Tensor, lo_hz: float, hi_hz: float) -> torch.Tensor:
    """One-sided periodogram band power (normalisation-agnostic) per row of (B, T)."""
    if sig.dim() == 1:
        sig = sig[None]
    spec = torch.fft.rfft(sig, dim=-1).abs() ** 2
    n = sig.shape[-1]
    return spec[..., int(round(lo_hz * n / 100)):int(round(hi_hz * n / 100))].sum(-1)


def _rule_f1(gate: dict, rule: str) -> float:
    v = gate[rule]
    if isinstance(v, dict):
        for k in ("f1", "F1", "f1_token"):
            if k in v:
                return float(v[k])
        raise AssertionError(f"fidelity_gate[{rule}] has no f1 entry: {v}")
    return float(v)


def _delta_band_index(bands) -> int:
    return next(i for i, (a, b) in enumerate(bands) if abs(a - 0.5) < 1e-6 and abs(b - 2.0) < 1e-6)


def _cycles(a: torch.Tensor) -> float:
    """Oscillation count of a burst atom: zero crossings of its visible part divided by 2."""
    sup = a[a.abs() > 0.15 * a.abs().max()]
    return float((sup[1:] * sup[:-1] <= 0).sum()) / 2


# ---------------------------------------------------------------------------
# 1-3: dictionary, background estimation
# ---------------------------------------------------------------------------

def test_01_dictionary_constraints():
    dicts = pytest.importorskip("physioex.models.psg_gen.dicts")
    d = dicts.full_dictionary()
    is_b = d.cls == dicts.CLS_BURST
    is_t = d.cls == dicts.CLS_TRANSIENT
    is_s = d.cls == dicts.CLS_SLOWWAVE
    assert is_b.any() and is_t.any() and is_s.any()
    band_names = list(dicts.BURST_BANDS)
    # bursts: >= 3 cycles (zero-crossing count of the visible part) and concentration >= 0.8
    for i in torch.nonzero(is_b).flatten().tolist():
        assert _cycles(d.atoms[i]) >= 3.0 - 0.5
        band = dicts.BURST_BANDS[band_names[int(d.band[i])]]
        assert dicts.spectral_concentration(d.atoms[i], band) >= 0.8
    # transients: zero-mean, <= 1 s, >= 80 % energy below 5 Hz
    for i in torch.nonzero(is_t).flatten().tolist():
        sup = d.atoms[i][d.atoms[i].abs() > 1e-6]
        assert float(sup.mean().abs()) <= 1e-2  # zero mean on the support
        assert float(d.dur[i]) <= 1.0 + 0.1  # <= 1 s (90 % energy support, +- measurement)
        assert dicts.energy_below(d.atoms[i], 5.0) >= 0.8
    # slow waves: one cycle of 1-2.5 s
    assert ((d.dur[is_s] >= 1.0 - 0.1) & (d.dur[is_s] <= 2.5 + 0.1)).all()
    rep = dicts.overlap_report(d)
    assert rep["coh_burst_transient"] < 0.3
    assert rep["coh_burst_slowwave"] < 0.3


def test_02_q20_gamma_factor():
    background = pytest.importorskip("physioex.models.psg_gen.background")
    gen = torch.Generator().manual_seed(0)
    # stationary white noise: the long-run band log power is the truth
    # the correction targets log10 of the mean band power, not the mean of the per-segment logs
    true = (10 ** background.seg_band_logpow(torch.randn(1, 3000 * 20, generator=gen))[0, 15:-15]).mean(0).log10()
    x = torch.randn(48, T, generator=gen)
    est = background.background_estimate(x, None, correct=True)["logP"]  # (B, n_win, n_bands)
    # bias = systematic error, so averaged over epochs; one epoch alone has the q20 sampling
    # spread (one 1 Hz bin in 0.5-2 Hz) on top. Units are log10 (0.1 = 1 dB).
    bias = (est[:, 3:-3].mean(1).mean(0) - true).abs()  # (n_bands,)
    assert float(bias.max()) <= 0.1  # corrected q20 within 1 dB of the true power


def test_03_background_bias():
    background = pytest.importorskip("physioex.models.psg_gen.background")
    bi = _delta_band_index(background.BG_BANDS)
    B = 8
    bg = synth.epochs("pink", B, seed=1)  # the floor, RMS 1
    x = bg.clone()
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):  # dense N3: 12 slow waves per epoch, ~90 % of the time covered
        jit = torch.rand(11, generator=torch.Generator().manual_seed(10 + b)) * 0.1 - 0.05
        # 11 slow waves on a 2.3 s grid (~80 % of the epoch): rejection sampling of 12 onsets
        # 2.2 s apart in 27 s almost never terminates
        for t0 in (2.5 + 2.3 * torch.arange(11) + jit).tolist():
            synth.plant(x[b], synth.slow_wave(2.2), t0, amp=2.0)  # peak-to-peak 4 units
            lo, hi = synth.support_span(t0, 2.2, pad=0.15)
            mask[b, lo:hi] = True
    est_m = background.background_estimate(x, mask, correct=True)["logP"]
    est_u = background.background_estimate(x, None, correct=True)["logP"]
    true_d = background.seg_band_logpow(bg)[..., bi].mean(1)  # (B,) per-epoch floor
    masked_bias = (est_m[:, 3:-3, bi].mean(1) - true_d).abs()
    unmasked_bias = est_u[:, 3:-3, bi].mean(1) - true_d
    assert float(masked_bias.max()) < 1.5  # masked estimate tracks the floor in 0.5-2 Hz
    assert float(unmasked_bias.min()) > 3.0  # without the mask the slow waves bias it


# ---------------------------------------------------------------------------
# 4-7: inference on planted events
# ---------------------------------------------------------------------------

def test_04_planted_spindles():
    dicts = pytest.importorskip("physioex.models.psg_gen.dicts")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    B = 8
    x = synth.epochs("pink", B, seed=2)
    planted = []
    for b in range(B):
        for f in (11.0, 13.0, 15.0):
            for t0 in synth.random_onsets(2, seed=20 + 3 * b + int(f), lo=2.0, hi=28.0, min_gap=3.0):
                synth.plant(x[b], synth.gabor_spindle(f, cycles=8), t0, amp=2.0)
                planted.append((b, t0, f))
    d, thr = _dict_and_thresholds()
    dec = inference.decompose(x, d, thr, BUDGET)
    ev = dec["events"]
    sigma = list(dicts.BURST_BANDS).index("sigma")
    matched = 0
    for b, t0, f in planted:
        sel = (ev.epoch == b) & (ev.cls == dicts.CLS_BURST) & ((ev.onset - t0).abs() <= 0.3)
        if not sel.any():
            continue
        best = int((ev.onset[sel] - t0).abs().argmin())
        i = int(ev.atom[sel][best])
        if int(d.band[i]) == sigma and abs(float(d.freq[i]) - f) <= 1.0:
            matched += 1  # right band (sigma) and the planted frequency on the 1 Hz grid
    assert matched / len(planted) >= 0.8


def test_05_explaining_away():
    pytest.importorskip("physioex.models.psg_gen.dicts")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    B = 8
    x = synth.epochs("pink", B, seed=3)
    for b in range(B):
        t_a = synth.random_onsets(1, seed=30 + b, lo=4.0, hi=8.0)[0]
        synth.plant(x[b], synth.gabor_spindle(12.0, cycles=5), t_a, amp=1.5)
        synth.plant(x[b], synth.gabor_spindle(12.0, cycles=5), t_a + 0.4, amp=1.5)  # 0.4 s apart
        t_k = synth.random_onsets(1, seed=40 + b, lo=18.0, hi=24.0)[0]
        synth.plant(x[b], synth.k_complex(0.7), t_k, amp=2.0)
        synth.plant(x[b], synth.gabor_spindle(13.0, cycles=5), t_k + 0.35, amp=1.5)  # on the KC tail
    d, thr = _dict_and_thresholds()
    ev, _ = inference.sparse_events(x, d, thr, BUDGET, explain=True, nms=True)
    n_planted = B * 3
    assert len(ev) / n_planted <= 1.2
    ev_topk = inference.topk_per_token(x, d)
    assert len(ev_topk) / n_planted > 1.5


def test_06_absorption():
    pytest.importorskip("physioex.models.psg_gen.dicts")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    B = 8
    x = synth.epochs("pink", B, seed=4)  # pure 1/f epochs, no events
    d, thr = _dict_and_thresholds()
    dec = inference.decompose(x, d, thr, BUDGET)
    x_ev = inference.reconstruct(dec["events"], d, B, T, exclude_periodic=True)
    share = (x_ev ** 2).sum() / (x ** 2).sum()
    assert float(share) < 0.05  # the events do not absorb the background


def test_07_ecg():
    pytest.importorskip("physioex.models.psg_gen.dicts")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    B = 8
    x = synth.epochs("pink", B, seed=5)
    for b in range(B):
        x[b] += synth.ecg_train(bpm=60.0, amp=2.0)
    d, thr = _dict_and_thresholds()
    ev, _ = inference.sparse_events(x, d, thr, BUDGET)
    pf = inference.periodic_flag(ev)
    assert int(pf.sum()) >= B  # the 60 bpm train is flagged periodic (per epoch)
    c_in = inference.counts(ev, B, exclude_periodic=False)
    c_ex = inference.counts(ev, B, exclude_periodic=True)
    assert int(c_ex.sum()) < int(c_in.sum())
    assert int(c_in.sum() - c_ex.sum()) >= int(pf.sum())  # the flagged events are excluded


# ---------------------------------------------------------------------------
# 8-9: encode/decode/intervene, P1
# ---------------------------------------------------------------------------

def test_08_encode_shapes():
    model_mod = pytest.importorskip("physioex.models.psg_gen.model")
    dicts = pytest.importorskip("physioex.models.psg_gen.dicts")
    B, L = 4, 5
    d, thr = _dict_and_thresholds()
    m = model_mod.PSGGenBase(d, BUDGET, thr, sequence_length=L).eval()
    bg = synth.epochs("pink", B * L, seed=6)
    xc = bg[:B].clone()
    for b in range(B):
        for t0 in synth.random_onsets(8, seed=60 + b, lo=2.0, hi=28.0, min_gap=2.5):
            synth.plant(xc[b], synth.slow_wave(1.5), t0, amp=2.0)
        for t0 in synth.random_onsets(4, seed=70 + b, lo=2.0, hi=28.0, min_gap=3.0):
            synth.plant(xc[b], synth.gabor_spindle(13.0, cycles=8), t0, amp=2.5)
        t_a = synth.random_onsets(1, seed=80 + b, lo=2.5, hi=27.5)[0]
        synth.plant(xc[b], synth.alpha_run(10.0, 1.5), t_a, amp=1.5)
    x = bg.reshape(B, L, 1, T)
    x[:, L // 2] = xc[:, None]
    with torch.no_grad():
        dec = m.encode(x)
    assert dec["N"].shape == (B, 4)
    assert dec["E"].shape == (B, m.d_e)
    assert dec["P"].shape == (B, 60, m.d_p)
    w = m.decode(dec)
    w_ev, w_bg = m.decode(dec, parts=True)
    assert w.shape == (B, T)
    assert torch.allclose(w, w_ev + w_bg, atol=1e-4)
    n0 = len(dec["events"])
    dec2 = m.intervene(dec, "remove_class", cls=dicts.CLS_BURST)
    assert len(dec["events"]) == n0  # intervene never mutates the input
    assert len(dec2["events"]) <= n0
    w2 = m.decode(dec2, parts=True)[0]
    assert (w2.pow(2).sum(-1) > 0).all()  # the other classes survive
    sig_before = _band_power(w_ev, 11.0, 16.0)
    sig_after = _band_power(w2, 11.0, 16.0)
    assert float(sig_before.max()) > 0
    assert (sig_after <= 1e-6 * w2.pow(2).sum(-1)).all()  # the sigma band of the events is zero
    with torch.no_grad():
        x_hat = m.decode(dec)
    gate = model_mod.fidelity_gate(xc, x_hat)
    assert _rule_f1(gate, "swa") >= 0.9
    assert _rule_f1(gate, "spindle") >= 0.9


def test_09_p1_fit_transients():
    p1 = pytest.importorskip("physioex.models.psg_gen.p1")
    dicts = pytest.importorskip("physioex.models.psg_gen.dicts")
    B = 16
    x = synth.epochs("pink", B, seed=8)
    for b in range(B):
        for t0 in synth.random_onsets(3, seed=80 + b, lo=2.5, hi=27.5, min_gap=3.0):
            synth.plant(x[b], synth.k_complex(0.7), t0, amp=2.0)
    d, thr = _dict_and_thresholds()
    m = p1.PSGGenP1(d, BUDGET, thr)
    m.fit_transients([x], n_iter=2)
    d2 = m.d
    true = synth.k_complex(0.7).double()
    true = (true - true.mean())
    true = true / true.norm()  # unit L2, like the dictionary atoms
    is_t = d2.cls == dicts.CLS_TRANSIENT
    best = 0.0
    for i in torch.nonzero(is_t).flatten().tolist():
        a = d2.atoms[i].double()
        v = torch.zeros_like(a)
        off = a.numel() // 2 - true.numel() // 2
        v[off:off + true.numel()] = true
        best = max(best, float((a * v).abs()))  # both unit L2, centre-aligned
    assert best >= 0.95  # a learned transient atom matches the true KC shape


# ---------------------------------------------------------------------------
# 10: P2 and P3 training steps
# ---------------------------------------------------------------------------

def _psg_batch(x4: torch.Tensor, B: int, L: int) -> dict:
    return {
        "signals": {"EEG_0": x4[:, :, 0, :].contiguous()},
        "channel_order": ["EEG_0"],
        "labels": torch.zeros(B, L, dtype=torch.long),
        "subject": [{"id": f"s{i % 2}"} for i in range(B)],
        "epoch_indices": torch.arange(L)[None].repeat(B, 1),
    }


def test_10_p2_p3_step():
    p2 = pytest.importorskip("physioex.models.psg_gen.p2")
    p3 = pytest.importorskip("physioex.models.psg_gen.p3")
    d, thr = _dict_and_thresholds()
    B, L = 8, 5
    x = synth.epochs("pink", B * L, seed=9).reshape(B, L, 1, T)
    for b in range(B):  # plant events in the centre epoch
        xc = x[b, L // 2, 0]
        for t0 in synth.random_onsets(4, seed=90 + b, lo=2.5, hi=27.5, min_gap=3.0):
            synth.plant(xc, synth.slow_wave(1.5), t0, amp=2.0)
        for t0 in synth.random_onsets(2, seed=95 + b, lo=2.5, hi=27.5, min_gap=4.0):
            synth.plant(xc, synth.gabor_spindle(13.0, cycles=8), t0, amp=2.5)
    for mod, cls_name in ((p2, "P2"), (p3, "P3")):
        try:
            _train_steps(mod, cls_name, d, thr, x, B, L)
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                pytest.skip(f"{cls_name} requires CUDA")
            raise


def _train_steps(mod, cls_name: str, d, thr: torch.Tensor, x: torch.Tensor, B: int, L: int,
                 n_steps: int = 200):
    trainer_mod = pytest.importorskip("physioex.models.psg_gen.trainer")
    inference = pytest.importorskip("physioex.models.psg_gen.inference")
    torch.manual_seed(0)
    m = getattr(mod, f"PSGGen{cls_name}")(d, BUDGET, thr, sequence_length=L)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = _psg_batch(x, B, L)
    xc = batch["signals"]["EEG_0"][:, L // 2]
    first, losses, extra = None, [], None
    for _ in range(n_steps):
        opt.zero_grad()
        loss, acc, extra = trainer_mod.PSGGenTrainer._step(m, batch, None, torch.device("cpu"))
        assert torch.isfinite(loss)  # one step runs, the loss is finite
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        if first is None:
            first = lv
        losses.append(lv)
    ratios = {k: v for k, v in extra.items() if "ratio" in str(k).lower() or "share" in str(k).lower()}
    assert ratios, f"{cls_name}: the extra dict must log the term ratios, got {sorted(extra)}"
    assert abs(sum(float(v) for v in ratios.values()) - 1.0) <= 0.2
    if cls_name == "P2":
        with torch.no_grad():
            dec = m(x)["dec"]
            x_hat = inference.reconstruct(dec["events"], m.d, B, T, exclude_periodic=True)
            mask = inference.support_mask(dec["events"], m.d, B, T)
            err = ((xc - x_hat) ** 2)[mask].sum()
            var = ((xc - xc.mean()) ** 2)[mask].sum()
            r2 = 1 - err / var
        assert float(r2) > 0.8  # time-domain R^2 on the supports
    else:
        assert float(torch.tensor(losses[-20:]).mean()) < first  # NLL decreases
