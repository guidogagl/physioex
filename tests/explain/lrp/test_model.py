"""End-to-end whole-model LRP tests (physioex.explain.lrp.model).

Synthetic models mirror the real PhysioEx families (transformer + attention
pooling; filterbank + BiLSTM + attention pooling + BiGRU; channel mixer;
conv + BatchNorm; dict output with standalone cross-attention).  Custom blocks
are registered through :func:`register_lrp_adapter`, as a user would.  Bias-free
variants must conserve exactly (Σ R ≈ f); biased ones must be finite.
"""

import copy
import warnings

import pytest
import torch
import torch.nn as nn

pytest.importorskip("lxt.explicit", reason="requires the 'explain' extra (lxt>=2.0)")
pytest.importorskip("zennit")

from physioex.explain.lrp import (  # noqa: E402
    LRPAttentionLayer,
    LRPAttentionPooling,
    LRPChannelMixer,
    ModelLRP,
    audit_lrp_coverage,
    prepare_model_for_lrp,
    register_lrp_adapter,
)
from physioex.explain.lrp.model import _epsilon_rule_factory  # noqa: E402

# ε-LRP conserves up to O(ε/|z|) per layer; on whole models with near-zero
# activations this shows up as ~1e-4 absolute on logit units — well below any
# meaningful attribution scale, but above a 1e-5 atol.
RTOL, ATOL = 1e-3, 1e-3


# --- custom block clones (registered like a user would) --------------------


class AttentionPooling(nn.Module):
    def __init__(self, d, a=8):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(d, a), nn.Tanh(), nn.Linear(a, 1, bias=False))

    def forward(self, x):
        return (x * torch.softmax(self.attention(x), dim=1)).sum(dim=1)


class AttentionLayer(nn.Module):
    def __init__(self, hidden, a=8):
        super().__init__()
        self.W_omega = nn.Parameter(torch.randn(hidden, a) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(a) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(a) * 0.1)

    def forward(self, x, r_alphas=False):
        B, S, H = x.size()
        v = torch.tanh(x.reshape(B * S, H) @ self.W_omega + self.b_omega.reshape(1, -1))
        exps = torch.exp(v @ self.u_omega.reshape(-1, 1)).reshape(-1, S)
        alphas = (exps / exps.sum(1, keepdim=True)).reshape(B, S, 1)
        return (x * alphas).sum(1)


class Filterbank(nn.Module):  # like seqsleepnet.LearnableFilterbank: x @ (sigmoid(W)·S)
    def __init__(self, F, D):
        super().__init__()
        self.W = nn.Parameter(torch.randn(F, D))
        self.S = nn.Parameter(torch.rand(F, D), requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, torch.sigmoid(self.W) * self.S)


class _NoDropout(nn.Module):
    def forward(self, x, zero_emb, channels_acc):
        return x


class ChannelMixer(nn.Module):  # like protosleepnet.ChannelMixer (eval path)
    def __init__(self, C, d, n_classes, bias):
        super().__init__()
        self.modality_emb = nn.Embedding(C, d)
        nn.init.zeros_(self.modality_emb.weight)  # constant add = 0 keeps conservation exact
        self.mcy = nn.Linear(d, n_classes)
        self.dropout = _NoDropout()
        self.register_buffer("channels_acc", torch.zeros(C))
        self.mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.0, bias=bias), 1
        )
        self.attn_pool = nn.Linear(d, 1)

    def forward(self, x, zero_emb):
        BL, C, d = x.shape
        x = x + self.modality_emb(torch.arange(C, device=x.device)).unsqueeze(0)
        mcy = self.mcy(x)
        x = self.dropout(x, zero_emb, self.channels_acc)
        x = x + self.mixer(x)
        w = torch.softmax(self.attn_pool(x), dim=1)
        return (x * w).sum(dim=1), mcy


register_lrp_adapter(AttentionPooling, LRPAttentionPooling)
register_lrp_adapter(AttentionLayer, LRPAttentionLayer)
register_lrp_adapter(Filterbank, _epsilon_rule_factory)
register_lrp_adapter(ChannelMixer, LRPChannelMixer)


# --- synthetic models: (B, L, ...) -> (B, L, n_classes) ---------------------


class TransModel(nn.Module):
    def __init__(self, d=16, bias=True):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 32, batch_first=True, dropout=0.0, bias=bias), 2
        )
        self.pool = AttentionPooling(d)
        self.head = nn.Linear(d, 5, bias=bias)

    def forward(self, x):  # (B, L, T, d)
        B, L, T, d = x.shape
        return self.head(self.pool(self.enc(x.reshape(B * L, T, d)))).view(B, L, -1)


class SpecRNNModel(nn.Module):  # filterbank -> BiLSTM -> attention -> BiGRU -> head
    def __init__(self, F=20, D=8, h=8, bias=True):
        super().__init__()
        self.fb = Filterbank(F, D)
        self.lstm = nn.LSTM(D, h, batch_first=True, bidirectional=True, bias=bias)
        self.att = AttentionLayer(2 * h)
        self.gru = nn.GRU(2 * h, h, batch_first=True, bidirectional=True, bias=bias)
        self.head = nn.Linear(2 * h, 5, bias=bias)

    def forward(self, x):  # (B, L, T, F)
        B, L, T, F = x.shape
        e, _ = self.lstm(self.fb(x.reshape(B * L, T, F)))
        s = self.gru(self.att(e).view(B, L, -1))[0]
        return self.head(s)


class ResidualModel(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.p = nn.Linear(8, 16, bias=bias)
        self.f = nn.Linear(16, 16, bias=bias)
        self.h = nn.Linear(16, 5, bias=bias)

    def forward(self, x):  # (B, L, T, 8)
        B, L, T, _ = x.shape
        h = self.p(x.reshape(B * L, T, -1))
        h = h + self.f(h)  # plain residual in the model's own forward
        return self.h(h.mean(1)).view(B, L, -1)


class MixerModel(nn.Module):
    def __init__(self, C=3, d=16, bias=False):
        super().__init__()
        self.mixer = ChannelMixer(C, d, 5, bias)
        self.zero_emb = nn.Parameter(torch.zeros(1, d), requires_grad=False)
        self.head = nn.Linear(d, 5, bias=bias)

    def forward(self, x):  # (B, L, C, d)
        B, L, C, d = x.shape
        h, _ = self.mixer(x.reshape(B * L, C, d), self.zero_emb)
        return self.head(h).view(B, L, -1)


class DictModel(nn.Module):  # CoReSleep-like: standalone MHA (tuple/kwargs) + residual + dict
    def __init__(self, d=16, bias=False):
        super().__init__()
        self.proj = nn.Linear(8, d, bias=bias)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, bias=bias)
        self.head = nn.Linear(d, 5, bias=bias)

    def forward(self, x):  # (B, L, T, 8)
        B, L, T, _ = x.shape
        h = self.proj(x.reshape(B * L, T, -1))
        a, _ = self.attn(query=self.norm(h), key=h, value=h, need_weights=False)
        h = h + a
        return {"combined": self.head(h.mean(dim=1)).view(B, L, -1)}


class ConvModel(nn.Module):
    def __init__(self, bias=False, bn=False):
        super().__init__()
        self.c1 = nn.Conv1d(1, 4, 8, stride=2, bias=bias)
        self.bn = nn.BatchNorm1d(4) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.c2 = nn.Conv1d(4, 4, 5, stride=2, bias=bias)
        self.fc = nn.Linear(4 * 13, 5, bias=bias)

    def forward(self, x):  # (B, L, 1, 64)
        h = self.act(self.bn(self.c1(x[:, 0])))
        return self.fc(self.c2(h).flatten(1)).unsqueeze(1)


def _report(model, x, **kw):
    rel, rep = ModelLRP(model, out_index=2, **kw)(x, return_report=True)
    return rel, rep


# --- forward equivalence ------------------------------------------------------


class TestPrepareForwardEquivalence:
    @pytest.mark.parametrize(
        "Model,shape",
        [(TransModel, (2, 3, 6, 16)), (SpecRNNModel, (2, 3, 6, 20)), (MixerModel, (2, 3, 3, 16))],
    )
    def test_prepared_forward_matches(self, Model, shape):
        torch.manual_seed(0)
        model = Model().eval()
        x = torch.randn(*shape)
        prep = prepare_model_for_lrp(copy.deepcopy(model))
        assert torch.allclose(prep(x), model(x), atol=1e-4)
        assert audit_lrp_coverage(prep) == []

    def test_prepare_is_idempotent(self):
        model = TransModel().eval()
        x = torch.randn(2, 3, 6, 16)
        prep = prepare_model_for_lrp(copy.deepcopy(model))
        again = prepare_model_for_lrp(prep)
        assert torch.allclose(again(x), model(x), atol=1e-4)

    def test_batchnorm_is_merged_and_forward_matches(self):
        torch.manual_seed(0)
        model = ConvModel(bias=False, bn=True)
        model.bn.running_mean.normal_()
        model.bn.running_var.uniform_(0.5, 2.0)
        model.bn.weight.data.uniform_(0.5, 1.5)
        model.bn.bias.data.normal_()
        model.eval()
        x = torch.randn(2, 3, 1, 64)
        prep = prepare_model_for_lrp(copy.deepcopy(model))
        assert torch.allclose(prep(x), model(x), atol=1e-4)
        assert type(prep.bn).__name__ == "IdentityRule"  # BN folded into c1 → identity
        assert audit_lrp_coverage(prep) == []


# --- conservation --------------------------------------------------------------


class TestConservation:
    @pytest.mark.parametrize(
        "Model,shape",
        [
            (TransModel, (2, 3, 6, 16)),
            (SpecRNNModel, (2, 3, 6, 20)),
            (MixerModel, (2, 3, 3, 16)),
            (ConvModel, (2, 3, 1, 64)),
        ],
    )
    def test_biasfree_models_conserve(self, Model, shape):
        torch.manual_seed(0)
        model = Model(bias=False).eval()
        x = torch.randn(*shape)
        rel, rep = _report(model, x)
        assert rel.shape == x.shape
        assert rep.is_conserved(RTOL, ATOL), rep.ratio

    def test_dict_output_and_standalone_mha_conserve(self):
        torch.manual_seed(0)
        rel, rep = _report(DictModel(bias=False).eval(), torch.randn(2, 3, 5, 8), output_key="combined")
        assert rep.is_conserved(RTOL, ATOL), rep.ratio

    def test_plain_residual_is_fixed_by_patch(self):
        torch.manual_seed(0)
        model, x = ResidualModel(bias=False).eval(), torch.randn(2, 3, 4, 8)
        _, patched = _report(model, x, patch_residuals=True)
        _, raw = _report(model, x, patch_residuals=False)
        assert patched.is_conserved(RTOL, ATOL), patched.ratio
        assert torch.allclose(raw.ratio, torch.full_like(raw.ratio, 2.0), rtol=1e-2)  # over-count

    def test_biased_model_finite_and_absorbs(self):
        rel, rep = _report(TransModel(bias=True).eval(), torch.randn(2, 3, 6, 16))
        assert torch.isfinite(rel).all() and torch.isfinite(rep.absorbed).all()


# --- robustness ----------------------------------------------------------------


class TestRobustness:
    def test_works_under_no_grad(self):
        with torch.no_grad():
            rel = ModelLRP(ResidualModel().eval(), out_index=1)(torch.randn(1, 2, 4, 8))
        assert torch.isfinite(rel).all()

    def test_shared_module_replaced_once(self):
        lin = nn.Linear(4, 4)
        m = nn.Module()
        m.a = lin
        m.b = lin
        prepare_model_for_lrp(m)
        assert m.a is m.b and type(m.a).__name__ == "EpsilonRule"

    def test_rank2_output(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 5, bias=False)

            def forward(self, x):
                return self.fc(x.flatten(1))

        rel, rep = ModelLRP(Flat().eval(), out_index=3)(torch.randn(3, 8), return_report=True)
        assert rel.shape == (3, 8) and rep.is_conserved(RTOL, ATOL)

    def test_original_model_untouched(self):
        model = TransModel().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ModelLRP(model)
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        assert all(p.requires_grad for p in model.parameters())

    def test_uncovered_custom_block_is_reported_and_strict_raises(self):
        class Weird(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                return x @ self.w

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.weird, self.fc = Weird(), nn.Linear(8, 5)

            def forward(self, x):
                return self.fc(self.weird(x))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prep = prepare_model_for_lrp(copy.deepcopy(Net()))
        assert audit_lrp_coverage(prep) == ["weird"] and any("weird" in str(m.message) for m in w)
        with pytest.raises(RuntimeError):
            prepare_model_for_lrp(Net(), strict=True)
