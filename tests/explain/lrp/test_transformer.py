"""Tests for instrumented AttnLRP (physioex.explain.lrp.transformer).

Two properties: **forward equivalence** to the fused ``nn.MultiheadAttention`` /
``nn.TransformerEncoderLayer``, and **relevance conservation** (Σ R ≈ f) on
bias-free layers (biases legitimately leak relevance in LRP, so conservation is
checked without them).  Attention uses the conservative-propagation (CP-LRP)
variant: relevance flows through the value path.

Skipped without the ``explain`` extra (``lxt``).
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("lxt", reason="requires the 'explain' extra (lxt)")

from physioex.explain.lrp.transformer import (  # noqa: E402
    LRPMultiheadAttention,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
    swap_transformer_layers,
)


def _conservation(fn, x, idx):
    x = x.detach().requires_grad_(True)
    out = fn(x)
    seed = torch.zeros_like(out)
    seed[idx] = out[idx].detach()
    out.backward(seed)
    return out[idx].detach(), x.grad.sum()


class TestMultiheadAttention:
    def test_forward_matches_nn_mha(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0).eval()
        x = torch.randn(2, 5, 16)
        ref = mha(x, x, x, need_weights=False)[0]
        got = LRPMultiheadAttention.from_torch(mha)(x, x, x)
        assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()

    def test_cross_attention_shape(self):
        mha = nn.MultiheadAttention(16, 4, batch_first=True).eval()
        lrp = LRPMultiheadAttention.from_torch(mha)
        q, kv = torch.randn(2, 3, 16), torch.randn(2, 7, 16)
        assert lrp(q, kv, kv).shape == (2, 3, 16)


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize("norm_first", [False, True])
    def test_forward_matches_nn_tel(self, norm_first):
        torch.manual_seed(0)
        tel = nn.TransformerEncoderLayer(
            16, 4, 32, batch_first=True, dropout=0.0, norm_first=norm_first
        ).eval()
        x = torch.randn(2, 5, 16)
        got = LRPTransformerEncoderLayer.from_torch(tel)(x)
        assert torch.allclose(got, tel(x), atol=1e-4), (got - tel(x)).abs().max()

    @pytest.mark.parametrize("norm_first", [False, True])
    def test_conservation_biasfree(self, norm_first):
        torch.manual_seed(0)
        tel = nn.TransformerEncoderLayer(
            16, 4, 32, batch_first=True, dropout=0.0,
            norm_first=norm_first, bias=False,
        ).eval()
        lay = LRPTransformerEncoderLayer.from_torch(tel)
        f, r = _conservation(lay, torch.randn(2, 5, 16), (0, 2, 3))
        assert torch.allclose(r, f, rtol=5e-2, atol=1e-3), (r, f)

    def test_finite_with_bias(self):
        tel = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0).eval()
        lay = LRPTransformerEncoderLayer.from_torch(tel)
        _, r = _conservation(lay, torch.randn(2, 5, 16), (0, 0, 1))
        assert torch.isfinite(r).all()


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0),
            num_layers=2,
        )

    def forward(self, x):
        return self.enc(x)


class TestSwap:
    def test_swap_replaces_encoder_and_matches_forward(self):
        torch.manual_seed(0)
        net = _Net().eval()
        x = torch.randn(2, 5, 16)
        ref = net(x)
        swap_transformer_layers(net)
        assert isinstance(net.enc, LRPTransformerEncoder)
        assert all(
            isinstance(l, LRPTransformerEncoderLayer) for l in net.enc.layers
        )
        assert torch.allclose(net(x), ref, atol=1e-4)
