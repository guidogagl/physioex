"""End-to-end model LRP tests (physioex.explain.lrp.model, Phase 2c).

Synthetic models mirror the two real PhysioEx families:
  * transformer + attention-pooling (à la SleepTransformer),
  * BiLSTM + attention-pooling + BiGRU (à la SeqSleepNet).

The custom pooling classes are named ``AttentionPooling`` / ``AttentionLayer``
so ``prepare_model_for_lrp`` matches and swaps them by name (as it does for the
real models).  Bias-free variants let us check exact conservation (Σ R ≈ f);
biases legitimately leak relevance in LRP.

Skipped without the ``explain`` extra (``lxt``).
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("lxt", reason="requires the 'explain' extra (lxt)")

from physioex.explain.lrp.model import ModelLRP, prepare_model_for_lrp  # noqa: E402


# --- custom pooling clones (matched by class name) -------------------------


class AttentionPooling(nn.Module):
    def __init__(self, d, a=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d, a), nn.Tanh(), nn.Linear(a, 1, bias=False)
        )

    def forward(self, x):
        w = torch.softmax(self.attention(x), dim=1)
        return (x * w).sum(dim=1)


class AttentionLayer(nn.Module):
    def __init__(self, hidden, a=8):
        super().__init__()
        self.W_omega = nn.Parameter(torch.randn(hidden, a) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(a) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(a) * 0.1)

    def forward(self, x, r_alphas=False):
        B, S, H = x.size()
        v = torch.tanh(x.reshape(B * S, H) @ self.W_omega + self.b_omega.reshape(1, -1))
        vu = v @ self.u_omega.reshape(-1, 1)
        exps = torch.exp(vu).reshape(-1, S)
        alphas = (exps / exps.sum(1, keepdim=True)).reshape(B, S, 1)
        return (x * alphas).sum(1)


# --- synthetic models: (B, L, T, d) -> (B, L, n_classes) -------------------


class TransModel(nn.Module):
    def __init__(self, d=16, heads=4, ff=32, n_classes=5, bias=True):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d, heads, ff, batch_first=True, dropout=0.0, bias=bias
            ),
            num_layers=2,
        )
        self.pool = AttentionPooling(d)
        self.head = nn.Linear(d, n_classes, bias=bias)

    def forward(self, x):  # x: (B, L, T, d)
        B, L, T, d = x.shape
        h = self.enc(x.reshape(B * L, T, d))
        h = self.pool(h)  # (B*L, d)
        return self.head(h).view(B, L, -1)


class RNNModel(nn.Module):
    def __init__(self, d=12, h=8, h2=8, n_classes=5, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True, bias=bias)
        self.att = AttentionLayer(2 * h)
        self.gru = nn.GRU(2 * h, h2, batch_first=True, bidirectional=True, bias=bias)
        self.head = nn.Linear(2 * h2, n_classes, bias=bias)

    def forward(self, x):  # x: (B, L, T, d)
        B, L, T, d = x.shape
        e, _ = self.lstm(x.reshape(B * L, T, d))
        e = self.att(e).view(B, L, -1)  # (B, L, 2h)
        s = self.gru(e)[0]  # (B, L, 2h2)
        return self.head(s)


def _conservation(model, x, in_index=0, out_index=2):
    lrp = ModelLRP(model, in_index=in_index, out_index=out_index)
    rel = lrp(x)
    with torch.no_grad():
        f = model.eval()(x)[:, in_index, out_index]
    return f, rel.flatten(1).sum(dim=1), rel.shape


class TestPrepareForwardEquivalence:
    @pytest.mark.parametrize("Model,d", [(TransModel, 16), (RNNModel, 12)])
    def test_prepared_forward_matches(self, Model, d):
        import copy

        torch.manual_seed(0)
        model = Model().eval()
        x = torch.randn(2, 3, 6, d)
        ref = model(x)
        prep = prepare_model_for_lrp(copy.deepcopy(model))
        assert torch.allclose(prep(x), ref, atol=1e-4), (prep(x) - ref).abs().max()


class TestModelConservation:
    def test_transformer_model_conservation_biasfree(self):
        torch.manual_seed(0)
        model = TransModel(bias=False).eval()
        x = torch.randn(2, 3, 6, 16)
        f, relsum, shape = _conservation(model, x)
        assert shape == x.shape
        assert torch.allclose(relsum, f, rtol=5e-2, atol=1e-3), (relsum, f)

    def test_rnn_model_conservation_biasfree(self):
        torch.manual_seed(0)
        model = RNNModel(bias=False).eval()
        x = torch.randn(2, 3, 6, 12)
        f, relsum, shape = _conservation(model, x)
        assert shape == x.shape
        assert torch.allclose(relsum, f, rtol=5e-2, atol=1e-3), (relsum, f)

    def test_relevance_finite_with_bias(self):
        model = TransModel(bias=True).eval()
        x = torch.randn(2, 3, 6, 16)
        rel = ModelLRP(model, out_index=1)(x)
        assert rel.shape == x.shape and torch.isfinite(rel).all()


class DictModel(nn.Module):
    """CoReSleep-like: a standalone ``nn.MultiheadAttention`` used with a tuple
    unpack + keyword args, and a dict output."""

    def __init__(self, d=16, heads=4, n_classes=5):
        super().__init__()
        self.proj = nn.Linear(8, d)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):  # (B, L, T, 8)
        B, L, T, _ = x.shape
        h = self.proj(x.reshape(B * L, T, -1))
        a, _ = self.attn(query=self.norm(h), key=h, value=h, need_weights=False)
        h = h + a
        return {"combined": self.head(h.mean(dim=1)).view(B, L, -1)}


class TestDictAndStandaloneMHA:
    def test_dict_output_and_mha_adapter_wire_up(self):
        torch.manual_seed(0)
        model = DictModel().eval()
        x = torch.randn(2, 3, 5, 8)
        rel = ModelLRP(model, out_index=1, output_key="combined")(x)
        assert rel.shape == x.shape and torch.isfinite(rel).all()
