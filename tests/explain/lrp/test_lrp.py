"""Unit tests for physioex.explain.lrp (Phase 1: core + CNN classics).

These are light, CPU-only tests over tiny synthetic models that honour the
PhysioEx ``(B, L, C, T) -> (B, L, n_classes)`` contract.  They validate the
LRP API, shape/finiteness, relevance conservation (the defining LRP property),
target selection, and the BatchNorm canonizer.

Heavy per-architecture checks over the real model zoo (Tsinalis, Chambon2018,
tinysleepnet, ...) run on Sofia, not here.

The whole module is skipped when the optional ``explain`` extra (``zennit``)
is not installed.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("zennit", reason="requires the 'explain' extra (zennit)")

from physioex.explain.lrp import (  # noqa: E402
    LRP,
    default_canonizers,
    epsilon_composite,
    physioex_composite,
)

C, T, N_CLASSES = 1, 64, 5


# ---------------------------------------------------------------------------
# Tiny synthetic models honouring (B, L, C, T) -> (B, 1, n_classes)
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Bias-free ReLU MLP — ε-LRP conserves relevance exactly onto the input."""

    def __init__(self, n_in=C * T, n_classes=N_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 16, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes, bias=False)

    def forward(self, x):
        h = x[:, 0].flatten(1)  # explain central epoch 0
        return self.fc2(self.act(self.fc1(h))).unsqueeze(1)


class TinyCNN(nn.Module):
    """Conv1d + BatchNorm + Linear — exercises the γ/w² composite + canonizer."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Conv1d(C, 4, kernel_size=8, stride=2)  # -> (B, 4, 29)
        self.bn = nn.BatchNorm1d(4)
        self.act = nn.ReLU()
        self.fc = nn.Linear(4 * 29, n_classes)

    def forward(self, x):
        h = self.act(self.bn(self.conv(x[:, 0])))
        return self.fc(h.flatten(1)).unsqueeze(1)


def _input(B=2, L=3):
    return torch.randn(B, L, C, T)


# ---------------------------------------------------------------------------
# API / shape / finiteness
# ---------------------------------------------------------------------------


class TestApi:
    def test_relevance_shape_matches_input(self):
        model = TinyCNN().eval()
        x = _input()
        rel = LRP(model, out_index=0)(x)
        assert rel.shape == x.shape

    def test_relevance_finite(self):
        model = TinyCNN().eval()
        rel = LRP(model, out_index=1)(_input())
        assert torch.isfinite(rel).all()

    def test_unbatched_input_gets_batch_dim(self):
        model = TinyCNN().eval()
        x = torch.randn(3, C, T)  # (L, C, T)
        rel = LRP(model)(x)
        assert rel.shape == (1, 3, C, T)

    def test_noncontributing_epochs_get_zero_relevance(self):
        # The models read only epoch 0; epochs 1.. must receive ~0 relevance.
        model = TinyMLP().eval()
        rel = LRP(model, composite=epsilon_composite())(_input(B=1, L=4))
        assert torch.allclose(rel[:, 1:], torch.zeros_like(rel[:, 1:]), atol=1e-6)


# ---------------------------------------------------------------------------
# Conservation — the defining LRP property (Σ R ≈ f(x))
# ---------------------------------------------------------------------------


class TestConservation:
    def test_epsilon_conserves_on_biasfree_mlp(self):
        model = TinyMLP().eval()
        x = _input(B=3, L=2)
        out_index = 2

        with torch.no_grad():
            logits = model(x)[:, 0, out_index]  # (B,)

        rel = LRP(model, out_index=out_index, composite=epsilon_composite())(x)
        rel_sum = rel.flatten(1).sum(dim=1)  # (B,)

        assert torch.allclose(rel_sum, logits, rtol=5e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


class TestTargetSelection:
    def test_different_classes_give_different_relevance(self):
        model = TinyCNN().eval()
        x = _input()
        r0 = LRP(model, out_index=0)(x)
        r1 = LRP(model, out_index=1)(x)
        assert not torch.allclose(r0, r1)


# ---------------------------------------------------------------------------
# Composites / canonizers
# ---------------------------------------------------------------------------


class TestComposites:
    def test_default_composite_with_batchnorm_runs(self):
        # BatchNorm present -> the SequentialMergeBatchNorm canonizer must kick in.
        model = TinyCNN().eval()
        rel = LRP(model)(_input())  # default physioex_composite + canonizer
        assert torch.isfinite(rel).all()

    def test_zbox_first_rule_builds_and_runs(self):
        model = TinyCNN().eval()
        comp = physioex_composite(
            first_rule="zbox", zbox_low=-5.0, zbox_high=5.0,
            canonizers=default_canonizers(model),
        )
        rel = LRP(model, composite=comp)(_input())
        assert rel.shape == (2, 3, C, T)

    def test_invalid_first_rule_raises(self):
        with pytest.raises(ValueError):
            physioex_composite(first_rule="bogus")
