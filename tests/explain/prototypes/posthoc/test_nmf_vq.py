"""Unit tests for physioex.explain.prototypes.posthoc NMF + VQ prototypes."""
import numpy as np
import pytest
import torch

from physioex.explain.prototypes.posthoc import (
    discover_prototypes_nmf,
    learn_codebook_kmeans,
    quantize_embeddings,
    VQBottleneck,
)


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    D = 8
    # two well-separated clusters -> classes 0 and 1
    z0 = rng.random((40, D)).astype(np.float32)
    z1 = rng.random((40, D)).astype(np.float32) + 5.0
    Z = np.concatenate([z0, z1])
    Y = np.concatenate([np.zeros(40), np.ones(40)]).astype(np.int64)
    return Z, Y


# ── NMF ──────────────────────────────────────────────────────────────

def test_discover_prototypes_nmf_shapes(embeddings):
    Z, Y = embeddings
    protos, labels = discover_prototypes_nmf(Z, Y, k_per_class=3, n_classes=2)
    assert protos.shape == (6, Z.shape[1])   # 3 per class * 2 classes
    assert labels.shape == (6,)
    assert set(labels.tolist()) == {0, 1}
    assert (labels[:3] == 0).all() and (labels[3:] == 1).all()


def test_discover_prototypes_nmf_deterministic(embeddings):
    Z, Y = embeddings
    p1, l1 = discover_prototypes_nmf(Z, Y, k_per_class=2, n_classes=2, random_state=7)
    p2, l2 = discover_prototypes_nmf(Z, Y, k_per_class=2, n_classes=2, random_state=7)
    assert np.allclose(p1, p2)
    assert np.array_equal(l1, l2)


def test_discover_prototypes_nmf_few_samples(embeddings):
    Z, Y = embeddings
    # request more prototypes than samples in a class -> uses all samples
    Z2 = np.concatenate([Z[:2], Z[40:42]])
    Y2 = np.array([0, 0, 1, 1], dtype=np.int64)
    protos, labels = discover_prototypes_nmf(Z2, Y2, k_per_class=5, n_classes=2)
    # each class had only 2 samples -> 2 prototypes each
    assert protos.shape[0] == 4


# ── VQ codebook ──────────────────────────────────────────────────────

def test_learn_codebook_kmeans_shape_dtype(embeddings):
    Z, _ = embeddings
    cb = learn_codebook_kmeans(Z, n_prototypes=5)
    assert cb.shape == (5, Z.shape[1])
    assert cb.dtype == np.float32


def test_quantize_embeddings_identity_when_input_is_codebook():
    codebook = np.eye(4, dtype=np.float32) * 3.0
    Zq, assign = quantize_embeddings(codebook.copy(), codebook)
    assert np.array_equal(assign, np.arange(4))
    assert np.allclose(Zq, codebook)


def test_quantize_embeddings_nearest():
    codebook = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    Z = np.array([[0.1, 0.1], [9.0, 9.0], [0.2, 0.0]], dtype=np.float32)
    Zq, assign = quantize_embeddings(Z, codebook)
    assert assign.tolist() == [0, 1, 0]
    assert np.allclose(Zq, codebook[assign])


# ── Differentiable VQ bottleneck ─────────────────────────────────────

def test_vq_bottleneck_forward_shape_and_loss():
    cb = np.random.default_rng(0).random((6, 4)).astype(np.float32)
    vq = VQBottleneck(cb, commitment_weight=0.25)
    z = torch.randn(10, 4, requires_grad=True)
    z_q, loss = vq(z)
    assert z_q.shape == z.shape
    assert loss.ndim == 0 and loss.item() >= 0.0


def test_vq_bottleneck_straight_through_gradient():
    cb = np.random.default_rng(1).random((6, 4)).astype(np.float32)
    vq = VQBottleneck(cb)
    z = torch.randn(5, 4, requires_grad=True)
    z_q, loss = vq(z)
    # STE: z_q = z + (z_q - z).detach() -> gradient of sum(z_q) wrt z is all ones
    z_q.sum().backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_vq_bottleneck_exact_when_z_is_codebook_entry():
    cb = (np.eye(4) * 2.0).astype(np.float32)
    vq = VQBottleneck(cb)
    z = torch.from_numpy(cb.copy())
    z_q, _ = vq(z)
    assert torch.allclose(z_q, z)
