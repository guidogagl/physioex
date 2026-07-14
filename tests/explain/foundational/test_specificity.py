"""Unit tests for physioex.explain.foundational.specificity strategies."""
import pytest
import torch

from physioex.explain.foundational.specificity import (
    SpecificityStrategy,
    CohenDSpecificity,
    MarginSpecificity,
    SoftmaxSpecificity,
    TopKSpecificity,
    NoFilter,
)


def test_base_strategy_is_abstract_by_convention():
    with pytest.raises(NotImplementedError):
        SpecificityStrategy().compute_mask(W=torch.zeros(5, 8))


# ── CohenDSpecificity (static) ───────────────────────────────────────

def test_cohen_d_requires_embeddings_labels_and_target():
    s = CohenDSpecificity()
    W = torch.randn(3, 4)
    with pytest.raises(ValueError):
        s.compute_mask(W)  # no embeddings/labels
    with pytest.raises(ValueError):
        s.compute_mask(W, embeddings=torch.randn(10, 4), labels=torch.zeros(10).long())


def test_cohen_d_highlights_separated_dimension():
    torch.manual_seed(0)
    D = 4
    # class 0 shifts dimension 0 strongly positive; others noise
    n = 50
    emb0 = torch.randn(n, D); emb0[:, 0] += 8.0
    emb1 = torch.randn(n, D)
    embeddings = torch.cat([emb0, emb1])
    labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()

    s = CohenDSpecificity(tau=1.0)
    mask = s.compute_mask(torch.randn(2, D), embeddings=embeddings, labels=labels, target_class=0)
    assert mask.shape == (D,)
    assert (mask >= 0).all() and (mask <= 1).all()
    # separated dim gets the highest specificity
    assert mask.argmax().item() == 0
    # cache: second call returns the same object
    assert s.compute_mask(torch.randn(2, D), embeddings=embeddings, labels=labels, target_class=0) is mask


# ── MarginSpecificity (dynamic) ──────────────────────────────────────

def test_margin_requires_x_and_target():
    s = MarginSpecificity()
    with pytest.raises(ValueError):
        s.compute_mask(W=torch.randn(3, 4))  # no x


def test_margin_shape_range_and_detached():
    s = MarginSpecificity(tau=1.0)
    W = torch.randn(3, 5)
    x = torch.randn(2, 5)
    mask = s.compute_mask(W, x=x, target_class=1)
    assert mask.shape == (2, 5)
    assert (mask >= 0).all() and (mask <= 1).all()
    assert not mask.requires_grad


# ── SoftmaxSpecificity (dynamic) ─────────────────────────────────────

def test_softmax_probabilities_sum_across_classes_to_one():
    tau = 1.0
    W = torch.randn(3, 6)
    x = torch.randn(4, 6)
    contrib = W.unsqueeze(0) * x.unsqueeze(1)
    probs_all = torch.softmax(contrib / tau, dim=1)  # (B, C, D)

    s = SoftmaxSpecificity(tau=tau)
    masks = [s.compute_mask(W, x=x, target_class=c) for c in range(3)]
    stacked = torch.stack(masks, dim=1)  # (B, C, D)
    assert torch.allclose(stacked.sum(dim=1), torch.ones(4, 6), atol=1e-5)
    assert torch.allclose(stacked, probs_all, atol=1e-6)


# ── TopKSpecificity ──────────────────────────────────────────────────

def test_topk_selects_exactly_k_dimensions():
    s = TopKSpecificity(k=3)
    W = torch.randn(3, 10)
    x = torch.randn(5, 10)
    mask = s.compute_mask(W, x=x, target_class=2)
    assert mask.shape == (5, 10)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(mask.sum(dim=-1) == 3)


# ── NoFilter ─────────────────────────────────────────────────────────

def test_nofilter_returns_ones_static_and_dynamic():
    s = NoFilter()
    W = torch.randn(4, 7)
    static = s.compute_mask(W)
    assert static.shape == (7,) and torch.all(static == 1.0)

    dynamic = s.compute_mask(W, x=torch.randn(3, 7))
    assert dynamic.shape == (3, 7) and torch.all(dynamic == 1.0)


def test_strategy_names():
    assert CohenDSpecificity().name == "cohen_d"
    assert MarginSpecificity().name == "margin"
    assert SoftmaxSpecificity().name == "softmax"
    assert TopKSpecificity().name == "topk"
    assert NoFilter().name == "none"
