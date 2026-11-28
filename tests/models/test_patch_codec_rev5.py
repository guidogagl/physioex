"""PatchCodec revision 5 (plan rev. 2 of xmai-lab/sleep-lens#17, T0): lapped patches with overlap-add synthesis,
the zero-phase spectral twin, the semantic group and the gated-attention MIL head of the sleep stage. CPU, small
models."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physioex.models.patch_codec import (
    EPOCH, PATCH, PatchCodec, PatchCodecTrainer, StageMIL, TransformCodec, build_from_args, ola_window, overlap_add,
    stage_bags, welch_psd, zero_phase_patches,
)


def _ar_noise(n_rows: int, n: int, seed: int = 0) -> torch.Tensor:
    """EEG-like coloured noise (as in test_patch_codec): AR(2) resonances at ~2 Hz and, weaker, ~11 Hz."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rows):
        lo = lfilter([1.0], [1.0, -2 * 0.97 * np.cos(2 * np.pi * 2 / 100), 0.97 ** 2], rng.standard_normal(n + 500))
        hi = lfilter([1.0], [1.0, -2 * 0.95 * np.cos(2 * np.pi * 11 / 100), 0.95 ** 2], rng.standard_normal(n + 500))
        out.append((lo + 0.3 * hi)[500:])
    x = torch.tensor(np.stack(out), dtype=torch.float32)
    return x / x.std()


M = 20  # margin of revision 5: 0.2 s
_LAPPED = {"quantizer": "fsq", "levels": 3, "fsq_levels": 5, "fsq_group": 2, "q_dropout": 0.0, "whiten": "signal",
           "dec_out": "white", "fsq_norm": True, "margin": M}


def _lapped(**kw):
    torch.manual_seed(0)
    m = PatchCodec(**{"width": 8, "dim": 16, "codes": 32, **_LAPPED, **kw})
    with torch.no_grad():
        m.whitener.fit_psd(*welch_psd(_ar_noise(4, 4000)))
    return m


def test_ola_window_ramps_sum_to_one():
    w = ola_window(M)
    assert w.shape == (PATCH + 2 * M,)
    torch.testing.assert_close(w[-2 * M:] + w[: 2 * M], torch.ones(2 * M))
    assert torch.all(w[2 * M:-2 * M] == 1) and torch.all((w > 0) & (w <= 1))


def test_overlap_add_is_the_identity_on_segments_of_one_signal():
    P = 7
    s = torch.randn(2, P * PATCH + 2 * M)
    y = s.unfold(1, PATCH + 2 * M, PATCH)  # segment i on samples 200 i - M .. 200 (i + 1) + M of the signal
    assert y.shape == (2, P, PATCH + 2 * M)
    torch.testing.assert_close(overlap_add(y, M), s[:, M:M + P * PATCH])
    t = torch.arange(P * PATCH + 2 * M) / 100.0
    sine = torch.sin(2 * torch.pi * 1.3 * t)[None]
    out = overlap_add(sine.unfold(1, PATCH + 2 * M, PATCH), M)
    torch.testing.assert_close(out, sine[:, M:M + P * PATCH])  # no step at the boundaries


def test_overlap_add_cross_fades_disagreeing_segments():
    y = torch.zeros(1, 2, PATCH + 2 * M)
    y[0, 1] = 1.0  # two constant segments that disagree by 1
    out = overlap_add(y, M)[0]
    assert out[: PATCH - M].abs().max() == 0 and torch.all(out[PATCH + M:] == 1)
    assert out.diff().abs().max() < 0.05  # a sin^2 ramp over 2 M samples, not a step of 1


@pytest.mark.parametrize("arch", ["conv", "skip"])
def test_lapped_codec_shapes_reach_and_chunked_coding(arch):
    from physioex.train.bin.patch_codec import decode_signal, encode_signal

    m = _lapped(arch=arch).eval()
    assert m.encoder_reach == 1 and m.receptive_patches >= 1
    x = torch.randn(2, 9 * PATCH)
    out = m(x)
    assert out["wave"].shape == (2, 9 * PATCH) and out["codes"].shape == (2, 9, 3)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
        z0 = m.encode(x[:1])
        x1 = x[:1].clone()
        x1[0, 5 * PATCH:5 * PATCH + 10] += torch.randn(10)  # start of patch 5: inside the margin of patch 4
        d = (m.encode(x1) - z0).abs().amax(-1)[0]
        assert d[4] > 0 and d[5] > 0 and torch.all(d[[0, 1, 2, 3, 6, 7, 8]] == 0)
        sig = torch.randn(23 * PATCH + 77)
        c1 = m.codes(sig[None])[0]
        assert torch.equal(encode_signal(m, sig, m.max_levels, chunk=4), c1)
        assert torch.allclose(decode_signal(m, c1, chunk=4), m.decode_codes(c1[None])[0], atol=1e-5)


def test_zero_phase_target_keeps_the_amplitude_spectrum():
    m = _lapped(zero_phase=True).eval()
    x = _ar_noise(1, 6 * PATCH, seed=3)
    t = m.white_target(x).view(6, PATCH)
    w = m.whitener(x).view(6, PATCH)
    torch.testing.assert_close(torch.fft.rfft(t.double()).abs(), torch.fft.rfft(w.double()).abs(), atol=1e-4, rtol=1e-4)
    k = torch.arange(1, PATCH // 2)
    torch.testing.assert_close(t[:, PATCH // 2 + k], t[:, PATCH // 2 - k], atol=1e-5, rtol=1e-5)  # symmetric pulse
    torch.testing.assert_close(t, zero_phase_patches(w))
    loss, r2, _ = m.loss(m(x), x)  # the codec reconstructs the twin's target
    assert torch.isfinite(loss) and torch.isfinite(r2)


def test_stage_bags_follow_the_patch_centres():
    g = torch.Generator().manual_seed(0)
    B, L, n = 64, 3, 6400
    o = torch.randint(0, L * EPOCH - n + 1, (B,), generator=g)
    labels = torch.randint(-1, 5, (B, L), generator=g)
    b, p0, y = stage_bags(o, n, labels)
    got = {(int(i), int(p), int(s)) for i, p, s in zip(b, p0, y)}
    want = set()
    for i in range(B):
        for e in range(L):
            if e * EPOCH >= o[i] and (e + 1) * EPOCH <= o[i] + n and 0 <= labels[i, e] < 5:
                p = int(np.ceil((e * EPOCH - int(o[i]) - PATCH // 2) / PATCH))
                want.add((i, p, int(labels[i, e])))
    assert got == want and len(got) > 0
    for i, p, s in got:  # the 15 patches of a bag have their centres in the epoch, inside the crop
        c = int(o[i]) + (p + np.arange(EPOCH // PATCH)) * PATCH + PATCH // 2
        e = c // EPOCH
        assert np.all(e == e[0]) and labels[i, e[0]] == s and p >= 0 and p + EPOCH // PATCH <= n // PATCH
    # a 64 s crop of a 90 s window always holds a whole epoch (scored or not)
    assert all(any(e * EPOCH >= o[i] and (e + 1) * EPOCH <= o[i] + n for e in range(L)) for i in range(B))


def test_stage_mil_finds_the_single_positive_instance():
    torch.manual_seed(0)
    K, d, nb = 15, 8, 256
    v = torch.zeros(d)
    v[0] = 3.0
    x = torch.randn(nb, K, d) * 0.5
    y = torch.arange(nb) % 2
    pos = torch.randint(0, K, (nb,))
    x[y == 1, pos[y == 1]] += v  # class 1: one instance carries the evidence
    head = StageMIL(d, n_classes=2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        logits, _ = head(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits, att = head(x)
    assert (logits.argmax(-1) == y).float().mean() > 0.95
    assert (att[y == 1].argmax(-1) == pos[y == 1]).float().mean() > 0.9


def _stage_batch(B=6, seed=0):
    """Windows of 3 epochs: signals ``(B, 3, 3000)`` (one channel), one stage per epoch."""
    g = torch.Generator().manual_seed(seed)
    return {"channel_order": ["EEG_clean"], "signals": {"EEG_clean": _ar_noise(B, 3 * EPOCH, seed).view(B, 3, EPOCH)},
            "labels": torch.randint(0, 5, (B, 3), generator=g)}


def test_stage_warmup_sets_lambda_then_the_encoder_sees_the_stage():
    m = _lapped(sem_dims=2, stage_weight=-0.1, stage_warmup=3, crop_sec=64.0).train()
    batch = _stage_batch()
    for step in range(4):
        m.zero_grad()
        loss, r2, extra = PatchCodecTrainer._step(m, batch, None, torch.device("cpu"))
        assert {"stage_ce", "stage_acc", "stage_bags"} <= set(extra) and torch.isfinite(loss)
        if step < 2:  # lambda is set at the end of the third (last) warm-up step
            assert float(m.stage_lambda) == 0.0
    lam = float(m.stage_lambda)
    assert lam > 0 and int(m.stage_steps) == 4
    # after the warm-up the cross-entropy reaches the encoder; before, only the head
    out = m(torch.randn(2, 6400))
    ce, w, _ = m.stage_loss(out["sem"], torch.zeros(2, dtype=torch.long), torch.randint(0, 5, (2, 3)), 6400)
    m.zero_grad()
    ce.backward()
    assert w == lam and m.encoder.out[-1].weight.grad.abs().sum() > 0


def test_stage_weight_zero_is_a_probe():
    m = _lapped(sem_dims=2, stage_weight=0.0, stage_warmup=1).train()
    for _ in range(3):
        out = m(torch.randn(2, 6400))
        ce, w, _ = m.stage_loss(out["sem"], torch.zeros(2, dtype=torch.long), torch.randint(0, 5, (2, 3)), 6400)
        m.zero_grad()
        ce.backward()
        assert w == 1.0 and float(m.stage_lambda) == 0.0
        assert m.encoder.out[-1].weight.grad is None or m.encoder.out[-1].weight.grad.abs().sum() == 0
        assert m.stage_head.cls.weight.grad.abs().sum() > 0


def test_build_from_args_revision5():
    a = {"width": 8, "dim": 16, "codes": 32, "levels": 4, "q_dropout": 0.0, "order": 32, "stft_weight": 0.0,
         "restart_after": 0, "crop_sec": 64.0, "quantizer": "fsq", "fsq_levels": 9, "fsq_group": 4,
         "whiten": "signal", "dec_out": "white", "arch": "skip", "fsq_norm": True, "ola_sec": 0.2, "zero_phase": True,
         "sem_dims": 8, "stage_weight": -0.1, "stage_warmup": 2000}
    m = build_from_args(a)
    assert m.margin == M and m.zero_phase and m.sem_dims == 8 and isinstance(m.stage_head, StageMIL)
    assert m(torch.randn(1, 6400))["sem"].shape == (1, 32, 8)
    old = {k: v for k, v in a.items() if k not in ("ola_sec", "zero_phase", "sem_dims", "stage_weight", "stage_warmup")}
    m0 = build_from_args(old)
    assert m0.margin == 0 and not m0.zero_phase and m0.stage_head is None


@pytest.mark.parametrize("groups", [1, 8])
def test_transform_codec_coefficients_are_the_drawn_patch_in_the_klt_basis(groups):
    torch.manual_seed(0)
    tc = TransformCodec(delta=0.35, groups=groups)
    with torch.no_grad():
        tc.whitener.fit_psd(*welch_psd(_ar_noise(4, 4000)))
        q, _ = torch.linalg.qr(torch.randn(PATCH, PATCH, dtype=torch.float64))
        tc.U.copy_(q.float())
    codes = tc.codes(_ar_noise(1, 5 * PATCH, seed=2))
    c = tc.coefficients(codes)
    assert c.shape == (1, 5, PATCH)
    torch.testing.assert_close((c @ tc.U.T).reshape(1, -1), tc._draw(tc.embed(codes)), atol=1e-4, rtol=1e-4)
