"""MSVQ (multi-scale residual VQ): shapes, trainer steps, main-sleep-period trim.

Random init, tiny batches, CPU.
"""
import numpy as np
import pytest
import torch

L = 5  # small odd window for speed; the default is 41


def _batch(B=4, n_epochs=50):
    torch.manual_seed(0)
    start = torch.randint(0, n_epochs - L, (B,))
    return {
        "signals": {"EEG_0": torch.randn(B, L, 3000)},
        "channel_order": ["EEG_0"],
        "labels": torch.zeros(B, L, dtype=torch.long),
        "subject": [{"id": f"s{i % 2}"} for i in range(B)],
        "epoch_indices": start[:, None] + torch.arange(L)[None],
    }


def _model(**kw):
    from physioex.models.msvq import MSVQ

    return MSVQ(sequence_length=L, dim=32, n_codes_n=4, n_codes_e=8, n_codes_p=8, **kw)


def test_msvq_forward_and_encode_shapes():
    model = _model().train()
    x = torch.randn(4, L, 1, 3000)
    out = model(x)
    assert out["k_n"].shape == (4,) and out["k_e"].shape == (4,) and out["k_p"].shape == (4, 30)
    assert out["spec_n"].shape == (4, 139) and out["spec_ne"].shape == (4, 139)
    assert out["mag_z"].shape == (4, 30, 51) and out["wave"].shape == (4, 30, 100)
    assert bool(model.vq_n.initialised)
    enc = model.eval().encode(x)
    assert enc["emb"].shape == (4, 3 * 32)


def test_msvq_n_ignores_centre_epoch():
    model = _model().train()
    model(torch.randn(16, L, 1, 3000))  # k-means init of the codebooks
    model.eval()
    x = torch.randn(2, L, 1, 3000)
    x2 = x.clone()
    x2[:, L // 2] = torch.randn(2, 1, 3000)
    a, b = model.quantize(x), model.quantize(x2)
    assert torch.equal(a["k_n"], b["k_n"]) and torch.allclose(a["s_n"], b["s_n"])


@pytest.mark.parametrize("quantize", [True, False])
def test_msvq_trainer_step_finite_and_backprop(quantize):
    from physioex.models.msvq import MSVQTrainer, fit_msvq_norms

    model = _model(quantize=quantize).train()
    fit_msvq_norms(model, [_batch()])
    loss, r2, extra = MSVQTrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.wave_head[0].weight.grad is not None and model.stem.proj.weight.grad is not None
    assert {"mse_spec_n", "phase", "mse_time", "r2_wave"} <= set(extra)
    assert ("perplexity_p" in extra) == quantize


def test_msvq_trainer_step_with_cluster_ce():
    from physioex.models.msvq import MSVQTrainer

    model = _model(n_targets_e=6, n_targets_p=6).train()
    rng = np.random.default_rng(0)
    model.targets = {
        "y_e": {s: rng.integers(0, 6, 50) for s in ("s0", "s1")},
        "y_p": {s: rng.integers(0, 6, (50, 30)) for s in ("s0", "s1")},
    }
    model.ce_weight = 0.1
    loss, _, extra = MSVQTrainer._step(model, _batch(), None, torch.device("cpu"))
    assert torch.isfinite(loss) and "ce" in extra


def test_night_descriptors_shapes_and_zscore():
    from physioex.models.msvq import night_descriptors

    sig = np.random.default_rng(0).standard_normal((40, 3000)).astype(np.float32)
    d_e, d_p = night_descriptors(sig, 5, 35)
    assert d_e.shape == (40, 6) and d_p.shape == (40, 30, 8)
    np.testing.assert_allclose(d_e[5:35].mean(0), 0, atol=1e-4)


def test_main_sleep_period_drops_nap_and_wake():
    from physioex.data.base import BasePhysioDataset

    y = np.zeros(2000, dtype=np.int16)
    y[100:140] = 2          # afternoon nap, 40 epochs
    y[800:1700] = 2         # main night
    y[1200:1300] = 0        # wake inside the night (< max_gap)
    y[1900:] = -1
    lo, hi = BasePhysioDataset._main_sleep_period(y, max_gap=120, margin=60)
    assert (lo, hi) == (740, 1760)
    assert BasePhysioDataset._main_sleep_period(np.zeros(10)) == (0, 10)
