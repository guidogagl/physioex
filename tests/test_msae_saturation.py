"""Penalty on saturated pre-tanh latents (``sat_weight``, EMG collapse 05.11.2026)."""
import torch

from physioex.models.msae import MSAE, tanh_saturation


def _emg_model():
    torch.manual_seed(0)
    return MSAE(3, d_p=8, token_sec=0.5, decoder="time", loss="mse", band_weight=1.0, fs=100, recon="texture",
                seg_sec=0.5, pow_bands=((10, 20), (20, 35), (35, 50)), psd_band=(10, 50), use_n=False)


def test_tanh_saturation_zero_inside_linear_region():
    assert tanh_saturation(torch.full((4, 8), 2.9)).item() == 0.0
    assert tanh_saturation(torch.full((4, 8), 5.0)).item() == 4.0


def _grad_e_out(model):
    x = torch.randn(2, 3, 1, 3000)
    model.train()
    model.encode(x)
    loss = model.sat_weight * model.last_sat
    model.zero_grad()
    loss.backward()
    return model.e_out.bias.grad


def test_penalty_restores_gradient_of_saturated_latents():
    m = _emg_model()
    with torch.no_grad():
        m.e_out.bias.fill_(20.0)  # every E latent deep in tanh saturation
    m.sat_weight = 0.1
    g = _grad_e_out(m)
    assert m.last_sat.item() > 0
    assert g is not None and (g > 0).all()  # descent lowers the bias: back into the linear region


def test_off_by_default():
    m = _emg_model()
    assert m.sat_weight == 0.0
    m.encode(torch.randn(2, 3, 1, 3000))
    assert not hasattr(m, "last_sat")
