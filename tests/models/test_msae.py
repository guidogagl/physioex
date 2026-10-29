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
