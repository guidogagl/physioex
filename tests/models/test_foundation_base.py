"""Contract tests for physioex.models.foundation_base.FoundationEncoder.

Uses a minimal concrete subclass (tiny nn.Linear, random init, no HuggingFace)
to exercise the wrapper logic: (B,L,C,T)->(B,L,D) reshaping, encoder freezing,
the encode() alias, the default _preprocess, and get_pipeline error handling.
"""
import pytest
import torch
import torch.nn as nn

from physioex.models.foundation_base import FoundationEncoder


class _FakeEncoder(FoundationEncoder):
    """Concrete encoder: mean over time then a linear projection to D=8."""

    MODEL_NAME = "fake"
    CHANNEL_STRATEGY = "all"
    # PIPELINE_PRESET intentionally left None to test get_pipeline error.

    def _build_encoder(self, in_chan, **kwargs):
        return nn.Linear(in_chan, 8)

    def _get_embedding_dim(self):
        return 8

    def _load_pretrained(self, checkpoint_path):
        return None  # random init

    def _encode(self, x):
        # x: (B, C, T) -> mean over T -> (B, C) -> Linear -> (B, 8)
        return self.encoder(x.mean(dim=-1))


@pytest.fixture
def model():
    return _FakeEncoder(in_chan=4)


def test_forward_reshapes_to_BLD(model):
    x = torch.randn(2, 5, 4, 100)  # (B, L, C, T)
    out = model(x)
    assert out.shape == (2, 5, 8)  # (B, L, D)


def test_encode_is_forward_alias(model):
    x = torch.randn(1, 3, 4, 50)
    assert torch.equal(model.encode(x), model(x))


def test_embedding_dim_and_in_chan(model):
    assert model.embedding_dim == 8
    assert model.in_chan == 4


def test_encoder_is_frozen(model):
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training  # eval() mode


def test_forward_runs_under_no_grad(model):
    x = torch.randn(1, 2, 4, 30)
    out = model(x)
    # embeddings come from a no_grad block -> not attached to autograd graph
    assert out.requires_grad is False


def test_default_preprocess_mean_centers(model):
    x = torch.randn(3, 4, 20) + 5.0
    out = model._preprocess(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(3, 4), atol=1e-4)


def test_get_pipeline_without_preset_raises():
    with pytest.raises(NotImplementedError):
        _FakeEncoder.get_pipeline()


# ── Abstract hooks must be implemented ───────────────────────────────

def test_base_hooks_raise_not_implemented():
    base = FoundationEncoder.__new__(FoundationEncoder)  # bypass __init__
    with pytest.raises(NotImplementedError):
        base._build_encoder()
    with pytest.raises(NotImplementedError):
        base._get_embedding_dim()
    with pytest.raises(NotImplementedError):
        base._load_pretrained(None)
    with pytest.raises(NotImplementedError):
        base._encode(torch.zeros(1, 1, 1))
