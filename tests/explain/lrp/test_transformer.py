"""Tests for the CP-LRP attention/transformer blocks (physioex.explain.lrp.transformer)."""

import pytest
import torch
import torch.nn as nn


from physioex.explain.lrp.transformer import (  # noqa: E402
    LRPMultiheadAttention,
    LRPMultiheadAttentionModule,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
    _cp_attention,
    swap_transformer_layers,
)


def _conservation(fn, x, idx):
    x = x.detach().requires_grad_(True)
    out = fn(x)
    seed = torch.zeros_like(out)
    seed[idx] = out[idx].detach()
    out.backward(seed)
    return out[idx].detach(), x.grad.sum()


class TestCPAttentionCore:
    def test_zero_relevance_to_q_k_and_exact_conservation(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 5, 8, requires_grad=True) for _ in range(3))
        out = _cp_attention(q, k, v, 0.35, 1e-6)
        # LRP seeding: output relevance is the output value (at a masked subset);
        # a random R independent of `out` would probe the ε/(out±ε) term instead.
        mask = (torch.rand_like(out) < 0.5).to(out.dtype)
        R = out.detach() * mask
        out.backward(R)
        assert q.grad is None or q.grad.abs().max() == 0
        assert k.grad is None or k.grad.abs().max() == 0
        assert torch.allclose(v.grad.sum(), R.sum(), rtol=1e-4, atol=1e-5)

    def test_half_precision_backward(self):
        q, k, v = (torch.randn(1, 2, 3, 4, dtype=torch.half, requires_grad=True) for _ in range(3))
        out = _cp_attention(q, k, v, 0.5, 1e-3)
        out.backward(torch.ones_like(out))
        assert v.grad.dtype == torch.half and torch.isfinite(v.grad).all()


class TestMultiheadAttention:
    def test_forward_matches_nn_mha(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0).eval()
        x = torch.randn(2, 5, 16)
        ref = mha(x, x, x, need_weights=False)[0]
        assert torch.allclose(LRPMultiheadAttention.from_torch(mha)(x, x, x), ref, atol=1e-4)

    def test_cross_attention_conservation_and_zero_query_path(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(16, 4, batch_first=True, bias=False).eval()
        lrp = LRPMultiheadAttention.from_torch(mha)
        q = torch.randn(2, 3, 16, requires_grad=True)
        kv = torch.randn(2, 7, 16, requires_grad=True)
        out = lrp(q, kv, kv)
        seed = torch.zeros_like(out)
        seed[0, 1, 2] = out[0, 1, 2].detach()
        out.backward(seed)
        assert q.grad.abs().max() == 0
        assert torch.allclose(kv.grad.sum(), out[0, 1, 2].detach(), rtol=1e-3, atol=1e-5)

    def test_multihead_consistency(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        for heads in (1, 2, 4):
            mha = nn.MultiheadAttention(16, heads, batch_first=True, bias=False).eval()
            f, r = _conservation(lambda z: LRPMultiheadAttention.from_torch(mha)(z, z, z), x, (0, 2, 3))
            assert torch.allclose(r, f, rtol=1e-3, atol=1e-5)

    def test_unsupported_configs_raise(self):
        with pytest.raises(NotImplementedError):
            LRPMultiheadAttention.from_torch(nn.MultiheadAttention(16, 4, batch_first=False))
        with pytest.raises(NotImplementedError):
            LRPMultiheadAttention.from_torch(nn.MultiheadAttention(16, 4, batch_first=True, add_bias_kv=True))

    def test_module_adapter_interface(self):
        mha = nn.MultiheadAttention(16, 4, batch_first=True).eval()
        ad = LRPMultiheadAttentionModule.from_torch(mha)
        x = torch.randn(2, 5, 16)
        out, w = ad(query=x, key=x, value=x, need_weights=True)
        assert out.shape == (2, 5, 16) and w.shape == (2, 5, 5)
        assert ad(x, x, x, need_weights=False)[1] is None
        with pytest.raises(NotImplementedError):
            ad(x, x, x, attn_mask=torch.zeros(5, 5, dtype=torch.bool))


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize("norm_first", [False, True])
    def test_forward_matches(self, norm_first):
        torch.manual_seed(0)
        tel = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0,
                                         norm_first=norm_first).eval()
        x = torch.randn(2, 5, 16)
        assert torch.allclose(LRPTransformerEncoderLayer.from_torch(tel)(x), tel(x), atol=1e-4)

    @pytest.mark.parametrize("norm_first", [False, True])
    @pytest.mark.parametrize("activation", ["relu", "gelu"])
    def test_conservation_biasfree(self, norm_first, activation):
        torch.manual_seed(0)
        tel = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0,
                                         norm_first=norm_first, bias=False,
                                         activation=activation).eval()
        f, r = _conservation(LRPTransformerEncoderLayer.from_torch(tel), torch.randn(2, 5, 16), (0, 2, 3))
        assert torch.allclose(r, f, rtol=1e-3, atol=1e-5), (r, f)

    def test_half_precision(self):
        tel = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0, bias=False).eval().half()
        lay = LRPTransformerEncoderLayer.from_torch(tel)
        x = torch.randn(2, 5, 16, dtype=torch.half, requires_grad=True)
        out = lay(x)
        out.backward(torch.ones_like(out))
        assert x.grad.dtype == torch.half and torch.isfinite(x.grad).all()

    def test_source_layer_not_frozen(self):
        tel = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True)
        LRPTransformerEncoderLayer.from_torch(tel)
        assert all(p.requires_grad for p in tel.parameters())

    def test_masks_raise(self):
        lay = LRPTransformerEncoderLayer.from_torch(nn.TransformerEncoderLayer(16, 4, 32, batch_first=True))
        with pytest.raises(NotImplementedError):
            lay(torch.randn(2, 5, 16), src_key_padding_mask=torch.zeros(2, 5, dtype=torch.bool))

    def test_seq_first_raises(self):
        with pytest.raises(NotImplementedError):
            LRPTransformerEncoderLayer.from_torch(nn.TransformerEncoderLayer(16, 4, 32, batch_first=False))


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, 32, batch_first=True, dropout=0.0), num_layers=2
        )

    def forward(self, x):
        return self.enc(x)


class TestSwap:
    def test_swap_replaces_encoder_matches_forward_and_is_idempotent(self):
        torch.manual_seed(0)
        net = _Net().eval()
        x = torch.randn(2, 5, 16)
        ref = net(x)
        swap_transformer_layers(net)
        swap_transformer_layers(net)  # idempotent
        assert isinstance(net.enc, LRPTransformerEncoder)
        assert torch.allclose(net(x), ref, atol=1e-4)
