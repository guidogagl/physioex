"""PatchCodec (2 s patch tokenizer, xmai-lab/sleep-lens#19): shapes, locality of encoder and decoder,
RVQ prefixes, FSQ and the context options of revision 2, chunked coding, whitening filter, crops, trainer step, continuity across patch boundaries. CPU, small models."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physioex.models.patch_codec import (
    FSQ, PATCH, PatchCodec, PatchContext, PatchCodecTrainer, Whitener, crops, has_zero_run, kmeans_init_rvq, levinson,
    warmup_continuous, welch_psd,
)

FS = 100


def _small(**kw):
    torch.manual_seed(0)
    return PatchCodec(width=8, dim=16, codes=32, levels=4, **kw)


def _ar_noise(n_rows: int, n: int, seed: int = 0) -> torch.Tensor:
    """EEG-like coloured noise: white noise through a resonant AR(2) at ~2 Hz plus a weaker one at ~11 Hz."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rows):
        w = rng.standard_normal(n + 500)
        lo = lfilter([1.0], [1.0, -2 * 0.97 * np.cos(2 * np.pi * 2 / FS), 0.97 ** 2], w)
        hi = lfilter([1.0], [1.0, -2 * 0.95 * np.cos(2 * np.pi * 11 / FS), 0.95 ** 2], rng.standard_normal(n + 500))
        out.append((lo + 0.3 * hi)[500:])
    x = torch.tensor(np.stack(out), dtype=torch.float32)
    return x / x.std()


def test_shapes_and_prefixes():
    m = _small().eval()
    x = torch.randn(2, 1650)  # cut to 8 patches
    out = m(x)
    assert out["wave"].shape == (2, 8 * PATCH)
    assert out["codes"].shape == (2, 8, 4)
    for k in (1, 2, 4):
        c = m.codes(x, levels=k)
        assert c.shape == (2, 8, k)
        assert m.decode_codes(c).shape == (2, 8 * PATCH)
    assert m.bits_per_patch(4) == 20


def test_quantizer_dropout_prefix_in_training():
    m = _small(q_dropout=1.0).train()
    ks = {m.quantize(m.encode(torch.randn(4, 800)))[1].shape[-1] for _ in range(40)}
    assert ks <= {1, 2, 3, 4} and len(ks) > 1


def test_encoder_is_local_to_the_patch():
    m = _small().eval()
    x = torch.randn(1, 6 * PATCH)
    z0 = m.encode(x)
    x1 = x.clone()
    x1[0, 2 * PATCH:3 * PATCH] += torch.randn(PATCH)
    z1 = m.encode(x1)
    changed = (z1 - z0).abs().amax(-1)[0]
    assert changed[2] > 0
    assert torch.all(changed[[0, 1, 3, 4, 5]] == 0)


@pytest.mark.parametrize("upsample", ["resize", "convT"])
def test_decoder_receptive_field_is_one_patch(upsample):
    """Output inside patch i depends on the codes of i - 1, i, i + 1 only (exact locality of edits)."""
    m = PatchCodec(width=8, dim=16, codes=32, levels=2, upsample=upsample).eval()
    with torch.no_grad():
        for p in m.decoder.parameters():  # break the ELU near-linearity of a fresh init a little
            p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(1, 9, 16)
        y0 = m.decode(z)
        z1 = z.clone()
        z1[0, 4] += torch.randn(16)
        d = (m.decode(z1) - y0).abs()[0].view(9, PATCH)
    per_patch = d.amax(1)
    assert torch.all(per_patch[[0, 1, 2, 6, 7, 8]] <= 1e-6 * per_patch.max())
    assert torch.all(per_patch[[3, 4, 5]] > 0)


def test_resize_upsampling_keeps_a_constant_constant():
    """A transposed convolution draws a pattern with the period of its stride from a constant input (the
    12.5 and 25 Hz lines of the pilot); repetition followed by a convolution does not."""
    from physioex.models.patch_codec import _up

    torch.manual_seed(0)
    x = torch.full((1, 4, 30), 0.7)
    with torch.no_grad():
        y_r = _up(4, 4, 4, (), "resize")(x)
        y_t = _up(4, 4, 4, (), "convT")(x)
    assert y_r.shape == y_t.shape == (1, 4, 120)
    assert y_r[..., 8:-8].std(-1).max() < 1e-6
    assert y_t[..., 8:-8].std(-1).max() > 1e-3


def test_line_ratios_on_and_off_the_grid():
    from physioex.train.bin.patch_codec import COMB_SEG, _line_ratios, _welch_sum

    g = torch.Generator().manual_seed(0)
    t = torch.arange(200_000) / FS
    x = torch.randn(200_000, generator=g) + 0.1 * torch.sin(2 * np.pi * 25 * t)
    p, c = _welch_sum(x)
    f = np.fft.rfftfreq(COMB_SEG, 1 / FS)
    on, off = _line_ratios(p / c, f), _line_ratios(p / c, f, offset=0.25)
    assert on[49] > 3 and off.max() < 2  # 25 Hz is the 50th line
    assert abs(np.median(on) - 1) < 0.05 and abs(np.median(off) - 1) < 0.05


def test_levinson_min_phase_and_whitening_flattens_the_spectrum():
    x = _ar_noise(64, 3000)
    w = Whitener(order=32)
    f, p = welch_psd(x)
    info = w.fit_psd(f, p)
    assert info["max_abs_reflection"] < 1
    assert np.all(np.abs(np.roots(w.a.double().numpy())) < 1)  # minimum phase
    e = w(x)
    f, pe = welch_psd(e[:, 32:])
    band = (f >= 1) & (f <= 30)
    raw_db, white_db = 10 * np.log10(p[band]), 10 * np.log10(pe[band])
    assert raw_db.std() > 5 and white_db.std() < 1.5


def test_levinson_matches_known_ar1():
    rho = 0.9
    r = rho ** np.arange(5)
    a, err, ks = levinson(r, 4)
    np.testing.assert_allclose(a, [1, -rho, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(err, 1 - rho ** 2, atol=1e-12)


def test_whitener_reflect_pad_is_causal_inside_the_patch():
    w = Whitener(order=4)
    w.a.copy_(torch.tensor([1.0, -0.5, 0.1, 0.0, 0.0]))
    x = torch.randn(3, 200)
    e = w(x)
    ref = x[:, 4:] - 0.5 * x[:, 3:-1] + 0.1 * x[:, 2:-2]
    torch.testing.assert_close(e[:, 4:], ref)


def test_crops_drop_zero_runs_and_are_centred_in_eval():
    x = torch.randn(3, 3000)
    x[1, 1000:2500] = 0
    assert has_zero_run(x).tolist() == [False, True, False]
    c = crops(x, 1600, training=False)
    assert c.shape == (2, 1600)
    torch.testing.assert_close(c[0], x[0, 700:2300])
    assert crops(x, 1600, training=True).shape[1] == 1600


def test_trainer_step_runs_and_backpropagates():
    m = _small(crop_sec=8.0).train()
    batch = {"channel_order": ["EEG_clean"], "signals": {"EEG_clean": _ar_noise(4, 3000)[:, None]}}
    loss, r2, extra = PatchCodecTrainer._step(m, batch, None, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert m.encoder.out[-1].weight.grad.abs().sum() > 0
    assert m.decoder.post[-1].weight.grad.abs().sum() > 0
    assert {"mse_w", "stft_w", "commit", "r2_raw", "perplexity_1"} <= set(extra)


def test_validation_loss_is_one_minus_r2():
    """The Trainer keeps the checkpoint and stops early on the lowest validation loss: for the codec that loss
    is 1 - whitened R^2, not the training loss with its commitment term."""
    m = _small(crop_sec=8.0).eval()
    batch = {"channel_order": ["EEG_clean"], "signals": {"EEG_clean": _ar_noise(4, 3000)[:, None]}}
    with torch.no_grad():
        loss, r2, extra = PatchCodecTrainer._step(m, batch, None, torch.device("cpu"))
    torch.testing.assert_close(loss, 1 - r2)
    assert "loss_train" in extra


def test_warmup_then_kmeans_initialises_every_level():
    m = _small(crop_sec=8.0)
    loader = [{"channel_order": ["EEG_clean"], "signals": {"EEG_clean": _ar_noise(4, 3000, seed=i)[:, None]}}
              for i in range(3)]
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    log = warmup_continuous(m, loader, opt, 5, torch.device("cpu"), log_every=2)
    assert [s for s, _ in log] == [2, 4, 5] and not m.bypass_vq and m.training
    assert all(not bool(vq.initialised) for vq in m.rvq.levels)
    n = kmeans_init_rvq(m, loader, torch.device("cpu"), per_code=1)  # 3 batches x 4 crops x 4 patches = 48
    assert n == 32 and m.training
    assert all(bool(vq.initialised) for vq in m.rvq.levels)
    m.eval()
    out = m(_ar_noise(2, 800, seed=9))
    assert out["codes"].shape == (2, 4, 4) and torch.isfinite(out["wave"]).all()


@pytest.mark.slow
def test_overfit_sinusoids_without_seams():
    """A small codec without VQ overfits sinusoids of random frequency and phase; the whitened error in
    the ±0.1 s around patch boundaries stays close to the error inside the patches."""
    torch.manual_seed(0)
    m = PatchCodec(width=16, dim=32, codes=32, levels=1, crop_sec=8.0)
    m.bypass_vq = True
    t = torch.arange(800) / FS
    g = torch.Generator().manual_seed(0)
    f = 1 + 14 * torch.rand(16, 1, generator=g)
    ph = 2 * np.pi * torch.rand(16, 1, generator=g)
    x = torch.sin(2 * np.pi * f * t + ph) + 0.5 * torch.sin(2 * np.pi * 0.7 * f * t)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        loss, r2, _ = m.loss(m(x), x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = (m.whitener(m(x)["wave"]) - m.whitener(x)).pow(2).mean(0)
    pos = torch.arange(800) % PATCH
    seam = (pos < 10) | (pos >= PATCH - 10)
    edge = (torch.arange(800) < PATCH) | (torch.arange(800) >= 800 - PATCH)  # crop edges see zero padding
    assert float(r2) > 0.9
    assert err[seam & ~edge].mean() <= 1.5 * err[~seam & ~edge].mean()


def test_subset_splits_same_number_per_item():
    from physioex.train.bin.patch_codec import subset_splits

    def unit(name, n):
        return {"train": [[name, f"{name}-{i}"] for i in range(n)], "val": [[name, "v"]], "test": [[name, "t"]]}

    sp = {"dcsm": unit("dcsm", 10), "hmc": unit("hmc", 3), "mass_ss02": unit("mass_ss02", 4),
          "mass_ss03": unit("mass_ss03", 6), "mesa": unit("mesa", 50)}
    sub, quota = subset_splits(sp, ["dcsm", "hmc", "mass"], 12, seed=0)
    assert quota == {"hmc": 3, "dcsm": 4, "mass": 5}  # ties keep the item order; the remainder goes last
    assert len(sub["mass_ss02"]["train"]) + len(sub["mass_ss03"]["train"]) == 5
    for k in sp:
        assert sub[k]["val"] == sp[k]["val"] and sub[k]["test"] == sp[k]["test"]
        assert all(x in sp[k]["train"] for x in sub[k]["train"])
        idx = [sp[k]["train"].index(x) for x in sub[k]["train"]]
        assert idx == sorted(idx)  # original order kept
    assert sub["mesa"] == sp["mesa"]  # not an item: untouched


# ---------------------------------------------------------------------------- revision 2 (27.11.2026)


def test_fsq_codes_round_trip_prefix_and_straight_through():
    q = FSQ(groups=3, group=2, levels=5)
    u = (1.5 * torch.randn(64, 6)).requires_grad_()
    v, c = q(u)
    assert c.shape == (64, 3) and int(c.min()) >= 0 and int(c.max()) < q.n_codes == 25
    assert set((v.detach() * q.half).round().unique().tolist()) == {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert torch.allclose(q.lookup(c), v.detach(), atol=1e-6)
    v.sum().backward()
    assert torch.all(u.grad[u.detach().abs() < 5] != 0)  # straight through the rounding (tanh slope only)
    v2, c2 = q(u.detach(), k=2)
    assert c2.shape == (64, 2) and torch.all(v2[:, 4:] == 0) and torch.equal(c2, c[:, :2])
    assert torch.allclose(q.lookup(c2), v2, atol=1e-6)


def test_fsq_codec_shapes_bits_and_decode_codes():
    torch.manual_seed(0)
    m = PatchCodec(width=8, dim=16, levels=3, quantizer="fsq", fsq_levels=5, fsq_group=2, q_dropout=0.0).eval()
    x = torch.randn(2, 1650)
    out = m(x)
    assert out["wave"].shape == (2, 8 * PATCH) and out["codes"].shape == (2, 8, 3)
    assert m.n_codes == 25 and m.bits_per_patch() == round(3 * 2 * math.log2(5))
    c = m.codes(x)
    assert torch.equal(c, out["codes"])
    assert torch.allclose(m.decode_codes(c), out["wave"], atol=1e-5)
    assert m.codes(x, levels=1).shape == (2, 8, 1)
    assert kmeans_init_rvq(m, [], "cpu") == 0  # no codebook


def test_patch_context_is_identity_at_init():
    z = torch.randn(2, 7, 16)
    assert torch.equal(PatchContext(16, 4)(z), z) and torch.equal(PatchContext(16, 0)(z), z)
    assert not list(PatchContext(16, 0).parameters())


def test_decoder_receptive_field_with_context():
    """ctx_dec = 2: the output inside patch i depends on the codes of i - 3 .. i + 3 only."""
    torch.manual_seed(0)
    m = PatchCodec(width=8, dim=16, codes=32, levels=2, ctx_dec=2).eval()
    assert m.receptive_patches == 3
    with torch.no_grad():
        for p in list(m.decoder.parameters()) + list(m.dec_ctx.parameters()):
            p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(1, 13, 16)
        y0 = m.decode(z)
        z1 = z.clone()
        z1[0, 6] += torch.randn(16)
        d = (m.decode(z1) - y0).abs()[0].view(13, PATCH).amax(1)
    assert torch.all(d[[0, 1, 2, 10, 11, 12]] <= 1e-6 * d.max())
    assert torch.all(d[3:10] > 0)


def test_encoder_reach_with_signal_whitening_and_context():
    """whiten="signal", ctx_enc = 2: the codes of patch i see patches i - 3 .. i + 2 (the whitening filter carries
    the last 32 samples of patch i - 1 into patch i)."""
    torch.manual_seed(0)
    m = PatchCodec(width=8, dim=16, codes=32, levels=2, whiten="signal", ctx_enc=2).eval()
    assert m.encoder_reach == 3
    with torch.no_grad():
        m.whitener.fit_psd(*welch_psd(_ar_noise(4, 4000)))
        for p in m.enc_ctx.parameters():
            p.add_(0.05 * torch.randn_like(p))
        x = torch.randn(1, 13 * PATCH)
        z0 = m.encode(x)
        x1 = x.clone()
        x1[0, 6 * PATCH:6 * PATCH + 50] += torch.randn(50)  # start of patch 6
        d1 = (m.encode(x1) - z0).abs().amax(-1)[0]
        x2 = x.clone()
        x2[0, 6 * PATCH - 10:6 * PATCH] += torch.randn(10)  # end of patch 5: reaches the start of patch 6
        d2 = (m.encode(x2) - z0).abs().amax(-1)[0]
    assert torch.all(d1[4:9] > 0) and torch.all(d1[[0, 1, 2, 3, 9, 10, 11, 12]] == 0)
    assert torch.all(d2[3:9] > 0) and torch.all(d2[[0, 1, 2, 9, 10, 11, 12]] == 0)


_FSQ_KW = {"quantizer": "fsq", "levels": 3, "fsq_levels": 5, "fsq_group": 2, "q_dropout": 0.0, "whiten": "signal"}


@pytest.mark.parametrize("kw", [{}, {**_FSQ_KW, "ctx_enc": 2, "ctx_dec": 2},
                                {**_FSQ_KW, "ctx_dec": 2, "dec_out": "white", "arch": "skip", "fsq_norm": True},
                                {**_FSQ_KW, "dec_out": "white", "arch": "linear", "fsq_norm": True}])
def test_chunked_coding_equals_one_pass(kw):
    from physioex.train.bin.patch_codec import decode_signal, encode_signal

    torch.manual_seed(0)
    m = PatchCodec(**{"width": 8, "dim": 16, "codes": 32, "levels": 2, **kw}).eval()
    with torch.no_grad():
        m.whitener.fit_psd(*welch_psd(_ar_noise(4, 4000)))
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
        sig = torch.randn(23 * PATCH + 77)
        c1 = m.codes(sig[None])[0]
        assert torch.equal(encode_signal(m, sig, m.max_levels, chunk=4), c1)
        assert torch.allclose(decode_signal(m, c1, chunk=4), m.decode_codes(c1[None])[0], atol=1e-5)


def test_inverse_whitener_round_trip():
    """gain / A(z) as a causal FIR of INV_TAPS samples undoes the whitening filter (zero history on both sides)."""
    w = Whitener()
    w.fit_psd(*welch_psd(_ar_noise(4, 8000)))
    w.gain.fill_(0.5)
    x = torch.randn(3, 3000)
    y = w.inverse(w(x, pad="zeros"))
    assert float((y - x).pow(2).mean() / x.pow(2).mean()) < 1e-6


def test_white_output_receptive_field_and_loss():
    """dec_out="white", arch="skip", ctx_dec = 2: the codes of patch i move the output of patches i - 3 .. i + 5
    only (the inverse FIR reaches 255 samples forward), within the declared receptive_patches = 5; the loss
    compares the drawn whitened patch with A * x."""
    torch.manual_seed(0)
    m = PatchCodec(width=8, dim=16, codes=32, levels=2, ctx_dec=2, dec_out="white", arch="skip").eval()
    assert m.receptive_patches == 5
    with torch.no_grad():
        m.whitener.fit_psd(*welch_psd(_ar_noise(4, 4000)))
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(1, 17, 16)
        y0 = m.decode(z)
        z1 = z.clone()
        z1[0, 8] += torch.randn(16)
        d = (m.decode(z1) - y0).abs()[0].view(17, PATCH).amax(1)
        assert torch.all(d[[0, 1, 2, 3, 4, 14, 15, 16]] <= 1e-6 * d.max())
        assert torch.all(d[5:13] > 0)
        x = torch.randn(2, 8 * PATCH)
        out = m(x)
        assert set(out) >= {"wave", "white"} and out["wave"].shape == out["white"].shape == x.shape
        loss, r2, _ = m.loss(out, x)
        assert torch.isfinite(loss)
        xw = m.whitener(x)
        assert torch.allclose(r2, 1 - (out["white"] - xw).pow(2).mean() / xw.var(), atol=1e-4)


def test_fsq_norm_keeps_the_bound_unsaturated():
    """Encoder latents far beyond the tanh bound (the dead start of pilot P2, sleep-lens#24): without the norm
    almost every input of the bound saturates; with it almost none does, and the codes still follow the latents."""
    sat = {}
    for norm in (False, True):
        torch.manual_seed(0)
        m = PatchCodec(width=8, dim=16, levels=2, quantizer="fsq", fsq_levels=5, fsq_group=2, q_dropout=0.0,
                       fsq_norm=norm).train()
        z = 1e3 * torch.randn(8, 6, 16)
        u = m.fsq_in(z)
        if norm:
            u = m.fsq_bn(u.reshape(-1, u.shape[-1])).view(u.shape)
        sat[norm] = float((torch.tanh(u).abs() > 0.99).float().mean())
        zq, codes, _ = m.quantize(z)
        assert codes.unique().numel() > 1 and torch.isfinite(zq).all()
    assert sat[False] > 0.9 and sat[True] < 0.05
