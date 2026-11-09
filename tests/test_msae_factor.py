"""Toy checks of the factorized encoder: shapes, the multiplicative decoder, the rule masks and
a few optimisation steps on random signals with two sites."""
import numpy as np
import pytest
import torch

from physioex.models.msae_factor import RULES, MSAEFactor, MSAEFactorTrainer, rule_masks


def _model(**kw):
    torch.manual_seed(0)
    return MSAEFactor(5, width=8, dim=16, n_atoms=8, sub=2, top_k=2, d_s=8, d_n=8, n_sites=2, **kw)


def test_shapes_and_decoder():
    m = _model().eval()
    x = torch.randn(3, 5, 1, 3000)
    z = m.encode(x)
    assert z["N"].shape == (3, 8) and z["E"].shape == (3, 8) and z["P"].shape == (3, 60, 16)
    assert (z["P"] >= 0).all() and ((z["P"] > 0).sum(-1) <= 2).all()
    torch.manual_seed(3)  # the background texture draws its noise: same seed, same texture
    w = m.decode(z["N"], z["E"], z["P"])
    assert w.shape == (3, 3000) and torch.isfinite(w).all()
    torch.manual_seed(3)
    det, bg = m.decode(z["N"], z["E"], z["P"], parts=True)
    assert torch.allclose(det + bg, w, atol=1e-4)
    # no atoms: only the background; no S: only the atoms
    torch.manual_seed(3)
    assert torch.allclose(m.decode(z["N"], z["E"], None), bg, atol=1e-4)
    assert torch.allclose(m.decode(z["N"], None, z["P"]), det, atol=1e-4)
    # the N gain scales the decoded: a different N changes the output of the same S and A
    m.gain_head.weight.data.normal_()  # zero at init: gain 1 for every N
    w2 = m.decode(torch.tanh(torch.randn(3, 8)), None, z["P"])
    assert not torch.allclose(w2, det)


def test_rule_masks_synthetic():
    fs, n = 100, 3000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, n)) * 5
    # epoch 0: a spindle at 10 s (13 Hz, 1 s, 30 uV) and a K-complex at 20 s (-150 uV, 0.7 s)
    x[0] += 30 * np.sin(2 * np.pi * 13 * t) * ((t > 10) & (t < 11))
    x[0] += -150 * np.sin(np.pi * (t - 20) / 0.7) * ((t > 20) & (t < 20.7))
    # epoch 1: alpha everywhere
    x[1] += 20 * np.sin(2 * np.pi * 10 * t)
    y = rule_masks(torch.from_numpy(x).float(), fs, 60)
    assert y.shape == (2, 60, 4)
    kc, swa, sp, al = (y[..., i] for i in range(4))
    assert sp[0, 20:22].any() and not sp[0, :15].any()
    assert kc[0, 40:42].any() and swa[0, 40:42].any() and not kc[0, 10:30].any()
    assert al[1].mean() > 0.8 and al[0].mean() < 0.3


def test_training_steps_decrease_loss():
    m = _model(l_rule=0.5)
    m.site_of_dataset = [0, 1]
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    torch.manual_seed(1)
    x = torch.randn(8, 5, 3000) * 20  # stack_channels adds the channel axis
    x[:4] += 40 * torch.sin(torch.arange(3000) / 100 * 2 * np.pi * 1.0)  # site 0: slow waves
    batch = {"signals": {"EEG_0": x}, "channel_order": ["EEG_0"], "labels": torch.zeros(8, 5, dtype=torch.long),
             "subject": [{"dataset_idx": int(i >= 4)} for i in range(8)]}
    losses = []
    for _ in range(12):
        loss, acc, extra = MSAEFactorTrainer._step(m, batch, None, torch.device("cpu"))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all() and losses[-1] < losses[0]
    assert {"pow", "mse_low", "bg", "sparse", "ctr", "site_a", "site_s", "rule"} <= set(extra)
