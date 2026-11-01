"""MSAE for other modalities (EOG, EMG): sampling rate, bands, texture-only decoder, no-N ablation."""
import torch

EMG_BANDS = ((10, 20), (20, 35), (35, 50), (50, 70), (70, 90))
EOG_BANDS = ((0.3, 1), (1, 2), (2, 5), (5, 9), (9, 15), (15, 35))


def _batch(x):
    return {"signals": {"EEG_0": x}, "channel_order": ["EEG_0"]}


def test_eeg_defaults_unchanged():
    from physioex.models.msae import MSAE

    m = MSAE(5, width=8, token_sec=0.5, d_p=16, texture=True)
    assert m.fs == 100 and m.epoch == 3000 and m.strides == (2, 5, 5) and m.n_tok == 60
    assert m.psd_mean.numel() == 139 and [c.kernel_size[0] for c in m.stem.branches] == [51, 25, 11, 5]


def test_200hz_same_tokens_and_texture_only():
    from physioex.models.msae import MSAE, MSAETrainer

    torch.manual_seed(0)
    m = MSAE(5, width=8, token_sec=0.5, d_p=4, fs=200, recon="texture", pow_bands=EMG_BANDS,
             psd_band=(10, 90), seg_sec=0.5)
    assert m.strides == (2, 2, 5, 5) and m.n_tok == 60 and not hasattr(m, "up")
    assert [c.kernel_size[0] for c in m.stem.branches] == [103, 51, 23, 11]
    x = torch.randn(3, 5, 6000)
    z = m(x[:, :, None])
    assert z["P"].shape == (3, 60, 4) and z["texture"].shape == (3, 6000) and z["wave_n"].shape == (3, 6000)
    assert m.band_logpow(x[:, 2]).shape == (3, 60, 5)
    loss, r2, extra = MSAETrainer._step(m, _batch(x), None, torch.device("cpu"))
    assert torch.isfinite(loss) and "r2_pow_ne" in extra


def test_texture_only_overfits_band_power():
    """Amplitude-modulated noise: the band power per 0.5 s token is learnable, loss falls."""
    from physioex.models.msae import MSAE, MSAETrainer

    torch.manual_seed(0)
    t = torch.linspace(0, 1, 6000)
    env = 1 + 3 * (torch.sin(6.28 * 4 * t) > 0.5).float()  # bursts
    x = torch.randn(4, 3, 6000) * env
    m = MSAE(3, width=8, dim=32, token_sec=0.5, d_p=4, fs=200, recon="texture", pow_bands=EMG_BANDS,
             psd_band=(10, 90), seg_sec=0.5)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    losses = []
    for _ in range(200):
        loss, _, _ = MSAETrainer._step(m, _batch(x), None, torch.device("cpu"))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    assert sum(losses[-10:]) / 10 < 0.5 * sum(losses[:10]) / 10, (losses[:3], losses[-3:])


def test_eog_bands_and_no_n():
    from physioex.models.msae import MSAE

    m = MSAE(5, width=8, token_sec=0.5, d_p=8, texture=True, pow_bands=EOG_BANDS, pow_hz=4.0, use_n=False).eval()
    assert [m.pow_bands[i] for i in m.tex_bands] == [(5, 9), (9, 15), (15, 35)]
    x = torch.randn(2, 5, 1, 3000)
    y = x.clone()
    y[:, [0, 1, 3, 4]] = torch.randn(2, 4, 1, 3000)  # other neighbours, same epoch t
    with torch.no_grad():
        a, b = m.encode(x), m.encode(y)
    assert torch.equal(a["N"], torch.zeros_like(a["N"]))
    assert torch.allclose(a["E"], b["E"]) and torch.allclose(a["P"], b["P"])


def test_vq_without_n_codes_and_bits():
    from physioex.models.msae_vq import MSAEVQ

    m = MSAEVQ(5, width=8, token_sec=0.5, d_p=4, fs=200, recon="texture", pow_bands=EMG_BANDS, psd_band=(10, 90),
               seg_sec=0.5, use_n=False, codes=16, codes_n=16, levels=(2, 2, 2)).eval()
    z = m.encode(torch.randn(2, 5, 1, 6000))
    assert z["codes"]["N"] is None and z["codes"]["E"].shape == (2, 2) and z["codes"]["P"].shape == (2, 60, 2)
    assert m.bits_per_epoch() == 2 * 4 + 60 * 2 * 4
    e = m.embed(z["codes"])
    assert torch.equal(e["N"], torch.zeros(2, m.dim)) and torch.allclose(e["E"], z["E"], atol=1e-5)


def test_modality_metrics_shapes_and_perfect_reconstruction():
    import numpy as np
    from physioex.models.msae import MSAE
    from physioex.models.recon_eval_modality import metrics

    rng = np.random.default_rng(0)
    night = np.repeat(np.arange(3), 20)
    for kw, n in ((dict(fs=200, recon="texture", pow_bands=EMG_BANDS, psd_band=(10, 90), seg_sec=0.5), 6000),
                  (dict(texture=True, pow_bands=EOG_BANDS, pow_hz=4.0), 3000)):
        m = MSAE(5, width=8, token_sec=0.5, d_p=4, **kw)
        x = rng.standard_normal((60, n)) * (1 + 3 * (rng.random((60, 1)) > 0.5))
        r = metrics(m, x, x, night)
        assert all(abs(v - 1) < 1e-6 for v in r["logpow5"].values()) and r["event_auc"] > 0.99
        assert len(r["per_night"]["event_auc"]) == 3
        assert ("env" in r) == (kw.get("recon") == "texture") and ("wave_low" in r) != ("env" in r)
