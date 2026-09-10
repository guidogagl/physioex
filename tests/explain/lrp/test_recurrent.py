"""Tests for the Arras signal-take LSTM/GRU LRP (physioex.explain.lrp.recurrent).

Two properties per module:
  1. **Forward equivalence** — the cell-level LRP module reproduces the fused
     ``nn.LSTM``/``nn.GRU`` output (LRP lives only in the backward pass).
  2. **Conservation** — with a bias-free recurrent module, total input relevance
     equals the seeded target output value (Σ R ≈ f).

Skipped without the ``explain`` extra (``lxt``).
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("lxt", reason="requires the 'explain' extra (lxt)")

from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM  # noqa: E402


def _seed_and_relevance(module, x, t, k):
    """Seed output relevance at ``out[:, t, k]`` (logit value) and return
    (target_value, total_input_relevance) per sample."""
    x = x.detach().requires_grad_(True)
    out = module(x)  # (B, T, dir*H)
    seed = torch.zeros_like(out)
    seed[:, t, k] = out[:, t, k].detach()
    out.backward(seed)
    return out[:, t, k].detach(), x.grad.flatten(1).sum(dim=1)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


class TestLRPLSTM:
    @pytest.mark.parametrize("bidirectional", [False, True])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_forward_matches_nn_lstm(self, bidirectional, num_layers):
        torch.manual_seed(0)
        lstm = nn.LSTM(
            input_size=6, hidden_size=5, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
        ).eval()
        lrp = LRPLSTM.from_torch(lstm).eval()
        x = torch.randn(3, 7, 6)
        with torch.no_grad():
            ref, _ = lstm(x)
            got = lrp(x)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()

    def test_conservation_biasfree(self):
        torch.manual_seed(1)
        lstm = nn.LSTM(4, 3, num_layers=1, batch_first=True, bias=False).eval()
        lrp = LRPLSTM.from_torch(lstm)
        x = torch.randn(2, 5, 4)
        target, relsum = _seed_and_relevance(lrp, x, t=4, k=1)
        assert torch.allclose(relsum, target, rtol=5e-2, atol=1e-3), (relsum, target)

    def test_relevance_finite_with_bias(self):
        lstm = nn.LSTM(4, 3, batch_first=True, bidirectional=True).eval()
        lrp = LRPLSTM.from_torch(lstm)
        x = torch.randn(2, 5, 4)
        _, relsum = _seed_and_relevance(lrp, x, t=0, k=2)
        assert torch.isfinite(relsum).all()


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------


class TestLRPGRU:
    @pytest.mark.parametrize("bidirectional", [False, True])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_forward_matches_nn_gru(self, bidirectional, num_layers):
        torch.manual_seed(0)
        gru = nn.GRU(
            input_size=6, hidden_size=5, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
        ).eval()
        lrp = LRPGRU.from_torch(gru).eval()
        x = torch.randn(3, 7, 6)
        with torch.no_grad():
            ref, _ = gru(x)
            got = lrp(x)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()

    def test_conservation_biasfree(self):
        torch.manual_seed(1)
        gru = nn.GRU(4, 3, num_layers=1, batch_first=True, bias=False).eval()
        lrp = LRPGRU.from_torch(gru)
        x = torch.randn(2, 5, 4)
        target, relsum = _seed_and_relevance(lrp, x, t=4, k=1)
        assert torch.allclose(relsum, target, rtol=5e-2, atol=1e-3), (relsum, target)
