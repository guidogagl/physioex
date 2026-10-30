"""MSAE (hierarchical continuous autoencoder) and recon_eval: shapes, conditional
chain, bit channel, trainer step, per-band R^2.

Random init, tiny batches, CPU.
"""
import numpy as np
import pytest
import torch

L = 5  # small odd window for speed; the default is 41


def _batch(B=2):
    torch.manual_seed(0)
    return {
        "signals": {"EEG_0": torch.randn(B, L, 3000)},
        "channel_order": ["EEG_0"],
        "labels": torch.zeros(B, L, dtype=torch.long),
        "subject": [{"id": f"s{i}"} for i in range(B)],
        "epoch_indices": torch.arange(L)[None].repeat(B, 1),
    }


def _model(**kw):
    from physioex.models.msae import MSAE

    torch.manual_seed(0)
    return MSAE(sequence_length=L, width=8, dim=32, **kw)


@pytest.mark.parametrize("token_sec,n_tok", [(1.0, 30), (0.5, 60)])
def test_msae_shapes(token_sec, n_tok):
    model = _model(d_p=4, token_sec=token_sec).eval()
    x = torch.randn(2, L, 1, 3000)
    out = model(x)
    assert out["N"].shape == (2, 32) and out["E"].shape == (2, 32) and out["P"].shape == (2, n_tok, 4)
    for k in ("wave_n", "wave_ne", "wave"):
        assert out[k].shape == (2, 3000)
    assert out["spec_n"].shape == (2, 139) and out["spec_ne"].shape == (2, 139)
    assert model.tokens(x).shape == (2, 2 + n_tok, 32)
    assert model.bits_per_epoch() == (32 + n_tok * 4) * 8


def test_n_does_not_depend_on_centre_epoch():
    model = _model().eval()
    x = torch.randn(2, L, 1, 3000)
    x2 = x.clone()
    x2[:, L // 2] = torch.randn(2, 1, 3000)
    with torch.no_grad():
        z, z2 = model.encode(x), model.encode(x2)
    assert torch.allclose(z["N"], z2["N"])
    assert not torch.allclose(z["E"], z2["E"])


def test_prefixes_differ_and_missing_latents_are_zero():
    model = _model().eval()
    with torch.no_grad():
        z = model.encode(torch.randn(2, L, 1, 3000))
        w_n, w_ne, w = model.decode(z["N"]), model.decode(z["N"], z["E"]), model.decode(z["N"], z["E"], z["P"])
        assert not torch.allclose(w_n, w_ne) and not torch.allclose(w_ne, w)
        assert torch.allclose(w_n, model.decode(z["N"], torch.zeros_like(z["E"]), torch.zeros_like(z["P"])), atol=1e-5)


def test_bit_channel_rounds_in_eval_and_adds_noise_in_training():
    from physioex.models.msae import _bottleneck

    z = torch.randn(1000)
    step = 2.0 / (2 ** 4 - 1)
    q = _bottleneck(z, 4, training=False)
    assert torch.allclose((q + 1) / step, ((q + 1) / step).round(), atol=1e-4)
    assert q.unique().numel() <= 16
    n = _bottleneck(z, 4, training=True)
    assert (n - torch.tanh(z)).abs().max() <= step / 2 + 1e-6
    assert torch.equal(_bottleneck(z, 0, training=False), torch.tanh(z))


def test_trainer_step_finite_and_backward():
    from physioex.models.msae import MSAETrainer

    model = _model().train()
    loss, acc, extra = MSAETrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss) and np.isfinite(acc.item())
    assert all(np.isfinite(v) for v in extra.values())
    loss.backward()
    assert model.down[0][0].block[1].weight.grad is not None
    assert model.dec_p.weight.grad is not None and model.enc_n.layers[0].linear1.weight.grad is not None


def test_band_r2_known_values():
    from physioex.models.recon_eval import band_r2, r2

    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100
    amp = lambda: np.repeat(rng.uniform(0.2, 2, 30), 100)  # random amplitude per second: amplitude R^2 needs variance
    y = np.stack([amp() * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6)) + amp() * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
                  + 0.1 * rng.standard_normal(3000) for _ in range(20)])
    assert r2(y, y) == 1.0
    lp = np.fft.irfft(np.where(np.arange(1501) <= 150, np.fft.rfft(y, axis=1), 0), n=3000, axis=1)  # <= 5 Hz
    b = band_r2(lp, y)
    assert b["phase"]["2-4"] > 0.99 and b["amp"]["2-4"] > 0.99
    assert b["phase"]["9-12"] < 0.1
    assert b["logpow"]["2-4"] > 0.99 and b["logpow"]["9-12"] < 0
    # a phase shift keeps the amplitude and loses the phase
    sh = np.roll(y, 5, axis=1)
    b = band_r2(sh, y)
    assert b["amp"]["9-12"] > 0.95 and b["phase"]["9-12"] < 0.5


def test_baselines_pca_full_rank_is_exact():
    from physioex.models.recon_eval import baselines

    rng = np.random.default_rng(0)
    basis = rng.standard_normal((4, 3000))
    tr, te = rng.standard_normal((50, 4)) @ basis, rng.standard_normal((10, 4)) @ basis
    out = baselines(tr, te, n_numbers=4, k_means=(8,))
    assert out["pca_4"]["r2"] > 0.999
    assert "kmeans_8_1s" in out


def test_tf_stft_istft_roundtrip():
    from physioex.models.msae import TF_BINS, TF_FRAMES, tf_istft, tf_stft

    x = torch.randn(3, 3000)
    s = tf_stft(x)
    assert s.shape == (3, TF_FRAMES, TF_BINS)
    assert torch.allclose(tf_istft(s), x, atol=1e-4)


def test_istft_decoder_shapes_and_mean_init():
    from physioex.models.msae import TF_BINS, TF_FRAMES

    model = _model(decoder="istft").eval()
    assert not hasattr(model, "up")
    model.lm_mean.fill_(-1.0)
    out = model(torch.randn(2, L, 1, 3000))
    for k in ("tf_n", "tf_ne", "tf"):
        assert out[k].shape == (2, TF_FRAMES, TF_BINS)
        assert torch.equal(out[k], torch.zeros_like(out[k]))  # mean log-spectrum at init
    assert out["phase"].shape == (2, TF_FRAMES, TF_BINS, 2) and out["wave"].shape == (2, 3000)
    assert "wave_n" not in out


def test_istft_trainer_step_finite_and_backward():
    from physioex.models.msae import MSAETrainer

    model = _model(decoder="istft").train()
    loss, acc, extra = MSAETrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in extra.values())
    loss.backward()
    assert model.mag_head.weight.grad.abs().sum() > 0 and model.phase_head.weight.grad is not None


def test_band_loss_helpers_and_step():
    from physioex.models.msae import MSAETrainer, band_logpow, lowpass_mse

    t = torch.arange(3000) / 100
    x = torch.stack([torch.sin(2 * np.pi * 2 * t), torch.sin(2 * np.pi * 11 * t)])
    bp = band_logpow(x)
    assert bp.shape == (2, 30, 7)
    assert bp[0, :, 1].min() > bp[0, :, 3].max() and bp[1, :, 3].min() > bp[1, :, 1].max()
    # low-pass MSE sees the 2 Hz error (= time MSE of a pure 2 Hz signal) and ignores 11 Hz
    assert torch.allclose(lowpass_mse(x[:1] * 0, x[:1]), x[:1].pow(2).mean(), rtol=1e-3)
    assert lowpass_mse(x[1:] * 0, x[1:]).item() < 1e-6
    model = _model(loss="band").train()
    loss, acc, extra = MSAETrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in extra.values())
    loss.backward()


def test_load_best_takes_newest_checkpoint(tmp_path):
    import os

    from physioex.models.recon_eval import best_checkpoint, load_best
    from physioex.train.trainer import Trainer

    assert best_checkpoint(str(tmp_path)) is None
    (tmp_path / "ckpt").mkdir()
    old, new = _model(), _model()
    torch.nn.init.zeros_(new.dec_p.weight)
    Trainer.save_checkpoint(old, str(tmp_path / "ckpt" / "a.pt"))
    Trainer.save_checkpoint(new, str(tmp_path / "ckpt" / "b.pt"))
    os.utime(tmp_path / "ckpt" / "a.pt", (1, 1))
    m = _model()
    assert load_best(m, str(tmp_path)).endswith("b.pt")
    assert torch.equal(m.dec_p.weight, new.dec_p.weight)


def test_band_loss_hybrid_full_mse():
    from physioex.models.msae import MSAETrainer

    model = _model(loss="band", low_hz=50, pow_hz=4.5).train()
    loss, _, extra = MSAETrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss) and np.isfinite(extra["pow_nep"])
    # low_hz=50 keeps every rfft bin: the low-pass MSE is the plain MSE
    from physioex.models.msae import lowpass_mse
    x, y = torch.randn(2, 3000), torch.randn(2, 3000)
    assert torch.allclose(lowpass_mse(x, y, 50), (x - y).pow(2).mean(-1), rtol=1e-3)


def test_texture_head_power_and_gradient_isolation():
    from physioex.models.msae import MSAETrainer, band_logpow

    model = _model(texture=True, pow_hz=4.5).eval()
    assert model.tex_bands == [2, 3, 4, 5, 6]
    out = model(torch.randn(2, L, 1, 3000))
    assert torch.allclose(out["wave"], out["wave_det"] + out["texture"])
    # gain 10^-3 at init: the texture is nearly silent; 9-13 Hz band of a 10^0 gain carries that band only
    assert out["texture"].abs().max() < 0.05
    with torch.no_grad():
        model.tex_head.bias.copy_(torch.tensor([-6.0, 0.0, -6.0, -6.0, -6.0]))
        bp = band_logpow(model(torch.randn(2, L, 1, 3000))["texture"])
    # 1 s rectangular segments leak across the 9 and 13 Hz edges: one decade of margin, on average
    assert bp[..., 3].mean() > bp[..., 2].mean() + 1 and bp[..., 3].mean() > bp[..., 4].mean() + 0.5
    # the band power loss never reaches the deterministic decoder path (up)
    model.train()
    loss, _, extra = MSAETrainer._step(model, _batch(), None, torch.device("cpu"))
    assert "pow_tex" in extra and torch.isfinite(loss)
    model.zero_grad()
    out = model(torch.stack([b for b in _batch()["signals"]["EEG_0"]])[:, :, None])
    (band_logpow(out["wave_det"].detach() + out["texture"])[..., model.tex_bands]).mean().backward()
    assert model.up[0].weight.grad is None or model.up[0].weight.grad.abs().sum() == 0
    assert model.tex_head.weight.grad.abs().sum() > 0


def test_band_r2_logpow5_forgives_random_phase():
    from physioex.models.recon_eval import band_r2

    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100
    env = lambda: np.repeat(10 ** rng.uniform(-0.4, 0.4, 6), 500)  # band envelope per 5 s
    noise = lambda: np.fft.irfft(np.where((np.arange(1501) >= 270) & (np.arange(1501) < 390),
                                          np.fft.rfft(rng.standard_normal(3000)), 0), n=3000)
    envs = [env() for _ in range(40)]
    y = np.stack([e * noise() for e in envs])
    pr = np.stack([e * noise() for e in envs])  # same envelope, independent phase
    b = band_r2(pr, y)
    assert b["phase"]["9-12"] < 0.1 and b["logpow5"]["9-12"] > 0.85 and b["logpow5"]["9-12"] > b["logpow"]["9-12"] + 0.1


def test_latent_probe_helpers():
    from physioex.models.latent_probe import _per_second, sigma_bursts, staging_probe

    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100
    x = 0.1 * rng.standard_normal((4, 3000))
    x[1, 500:600] += np.sin(2 * np.pi * 13 * t[500:600])  # one sigma burst in second 5 of epoch 1
    b = sigma_bursts(x, np.zeros(4, int), q=99)
    assert b[1, 5] and b.sum() <= 2
    z = {"E": rng.standard_normal((3, 8)), "P": rng.standard_normal((3, 60, 4))}
    ps = _per_second(z)
    assert ps["E"].shape == (90, 8) and ps["EP"].shape == (90, 16)
    assert np.allclose(ps["EP"][31, 8:], z["P"][1, 2:4].reshape(-1))  # second 1 of epoch 1 = tokens 2, 3
    y = rng.integers(0, 3, 300)
    f = {"good": np.eye(3)[y] + 0.01 * rng.standard_normal((300, 3)), "bad": rng.standard_normal((300, 3))}
    r = staging_probe(f, y, f, y, np.repeat([0, 1, 2], 100))
    assert r["good"]["kappa_night_mean"] > 0.95 and r["bad"]["kappa_night_mean"] < 0.3


def test_grid_shift_moves_only_in_training():
    from physioex.models.msae import band_logpow, grid_shift

    x = torch.randn(2, 3000)
    assert grid_shift(False, x)[0] is x
    torch.manual_seed(1)
    (g,) = grid_shift(True, x)
    assert g.shape == (2, 2900) and band_logpow(g).shape == (2, 29, 7)


def test_noninferiority_paired():
    from physioex.models.noninferiority import noninferiority

    rng = np.random.default_rng(0)
    ref = rng.uniform(0.9, 0.99, 39)
    same = noninferiority(ref, ref + rng.normal(0, 0.002, 39), margin=0.01)
    assert same["pass"] and same["p"] < 0.001 and same["lower_t"] > -0.01
    worse = noninferiority(ref, ref - 0.02 + rng.normal(0, 0.002, 39), margin=0.01)
    assert not worse["pass"] and worse["p"] > 0.5
    # few noisy nights: no significant difference, but non-inferiority not shown either
    noisy = noninferiority(ref[:5], ref[:5] + np.array([0.03, -0.03, 0.03, -0.03, 0.0]), margin=0.01)
    assert not noisy["pass"]


def test_night_metrics_and_sample_windows_night():
    from physioex.models.recon_eval import night_metrics

    rng = np.random.default_rng(0)
    y = rng.standard_normal((6, 3000)); night = np.array([0, 0, 0, 2, 2, 2])
    m = night_metrics(y, y, night)
    assert m["night"] == [0, 2] and m["r2"] == [1.0, 1.0] and len(m["logpow5"]["9-12"]) == 2
