"""Tests for the Arras signal-take LSTM/GRU LRP (physioex.explain.lrp.recurrent).

Forward equivalence to the fused ``nn.LSTM``/``nn.GRU`` (all configs), exact
conservation on bias-free stacks (incl. multi-layer bidirectional, seeding the
reverse half mid-sequence), and the interface guards.
"""

import pytest
import torch
import torch.nn as nn


from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM  # noqa: E402

CFGS = [(1, False), (1, True), (2, False), (2, True)]


def _conservation(module, x, t, k):
    x = x.detach().requires_grad_(True)
    out = module(x)[0]
    seed = torch.zeros_like(out)
    seed[:, t, k] = out[:, t, k].detach()
    out.backward(seed)
    return out[:, t, k].detach(), x.grad.flatten(1).sum(dim=1)


@pytest.mark.parametrize("num_layers,bidirectional", CFGS)
class TestLSTM:
    def test_forward_matches(self, num_layers, bidirectional):
        torch.manual_seed(0)
        lstm = nn.LSTM(6, 5, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional).eval()
        lrp = LRPLSTM.from_torch(lstm)
        x = torch.randn(3, 7, 6)
        with torch.no_grad():
            ref, (h_ref, c_ref) = lstm(x)
            got, (h, c) = lrp(x)
        assert torch.allclose(got, ref, atol=1e-5)
        assert torch.allclose(h, h_ref, atol=1e-5) and torch.allclose(c, c_ref, atol=1e-5)

    def test_conservation_biasfree(self, num_layers, bidirectional):
        torch.manual_seed(1)
        lstm = nn.LSTM(4, 3, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional, bias=False).eval()
        lrp = LRPLSTM.from_torch(lstm)
        k = 3 + 1 if bidirectional else 1  # seed the reverse half when present
        f, r = _conservation(lrp, torch.randn(2, 5, 4), t=2, k=k)
        assert torch.allclose(r, f, rtol=1e-3, atol=1e-5), (r, f)


@pytest.mark.parametrize("num_layers,bidirectional", CFGS)
class TestGRU:
    def test_forward_matches(self, num_layers, bidirectional):
        torch.manual_seed(0)
        gru = nn.GRU(6, 5, num_layers=num_layers, batch_first=True,
                     bidirectional=bidirectional).eval()
        lrp = LRPGRU.from_torch(gru)
        x = torch.randn(3, 7, 6)
        with torch.no_grad():
            ref, h_ref = gru(x)
            got, h = lrp(x)
        assert torch.allclose(got, ref, atol=1e-5) and torch.allclose(h, h_ref, atol=1e-5)

    def test_conservation_biasfree(self, num_layers, bidirectional):
        torch.manual_seed(1)
        gru = nn.GRU(4, 3, num_layers=num_layers, batch_first=True,
                     bidirectional=bidirectional, bias=False).eval()
        lrp = LRPGRU.from_torch(gru)
        k = 3 + 1 if bidirectional else 1
        f, r = _conservation(lrp, torch.randn(2, 5, 4), t=2, k=k)
        assert torch.allclose(r, f, rtol=1e-3, atol=1e-5), (r, f)


class TestInterface:
    def test_isinstance_and_tuple_interface(self):
        lrp = LRPGRU.from_torch(nn.GRU(4, 3, batch_first=True))
        assert isinstance(lrp, nn.GRU)
        out, h = lrp(torch.randn(2, 5, 4))
        assert out.shape == (2, 5, 3) and h.shape == (1, 2, 3)

    def test_epsilon_is_plumbed(self):
        assert LRPLSTM.from_torch(nn.LSTM(4, 3), epsilon=1e-3).epsilon == 1e-3
        assert LRPGRU.from_torch(nn.GRU(4, 3), epsilon=1e-3).epsilon == 1e-3

    def test_hx_raises(self):
        lrp = LRPLSTM.from_torch(nn.LSTM(4, 3, batch_first=True))
        with pytest.raises(NotImplementedError):
            lrp(torch.randn(2, 5, 4), (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)))

    def test_proj_size_raises(self):
        with pytest.raises(NotImplementedError):
            LRPLSTM.from_torch(nn.LSTM(4, 6, proj_size=3))

    def test_edge_shapes_forward(self):
        lstm = nn.LSTM(4, 3, batch_first=True).eval()
        lrp = LRPLSTM.from_torch(lstm)
        for shape in [(1, 1, 4), (1, 3, 4)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                assert torch.allclose(lrp(x)[0], lstm(x)[0], atol=1e-5)

    def test_biased_relevance_finite(self):
        lrp = LRPGRU.from_torch(nn.GRU(4, 3, batch_first=True, bidirectional=True).eval())
        _, r = _conservation(lrp, torch.randn(2, 5, 4), t=0, k=2)
        assert torch.isfinite(r).all()

    def test_source_model_untouched(self):
        lstm = nn.LSTM(4, 3, batch_first=True)
        LRPLSTM.from_torch(lstm)
        assert all(p.requires_grad for p in lstm.parameters())
