"""Accuracy-weighted modality dropout of ProtoSleepNet.

The drop probability of a channel must be proportional to its accuracy,
``p_c = a_c / sum_j a_j``: the most accurate channel is the one occluded
most often.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physioex.models.protosleepnet import ZeroEmbeddingDropout  # noqa: E402

pytestmark = pytest.mark.unit


def _drop_counts(acc, n_samples=4000, seed=0):
    torch.manual_seed(seed)
    nchan = len(acc)
    drop = ZeroEmbeddingDropout(p_apply=1.0).train()
    x = torch.ones(n_samples, nchan, 4)
    zero_emb = torch.zeros(1, 4)
    out = drop(x, zero_emb, torch.tensor(acc))
    dropped = (out == 0).all(dim=-1)  # (n_samples, nchan)
    return dropped.float().sum(dim=0)


def test_high_accuracy_channel_is_dropped_most():
    counts = _drop_counts([0.9, 0.3, 0.1])
    assert counts[0] > counts[1] > counts[2]


def test_drop_frequency_follows_accuracy_ratio():
    # With two channels, exactly one is dropped per sample (n in {1}), so
    # the empirical drop frequency equals p_c = a_c / sum_j a_j.
    counts = _drop_counts([0.8, 0.2], n_samples=20000)
    freq = counts / counts.sum()
    assert abs(freq[0].item() - 0.8) < 0.02
    assert abs(freq[1].item() - 0.2) < 0.02


def test_never_drops_all_channels():
    torch.manual_seed(0)
    drop = ZeroEmbeddingDropout(p_apply=1.0).train()
    x = torch.ones(512, 3, 4)
    out = drop(x, torch.zeros(1, 4), torch.tensor([0.5, 0.3, 0.2]))
    kept = (out != 0).any(dim=-1).sum(dim=1)
    assert (kept >= 1).all()


def test_uniform_fallback_without_accuracy_estimates():
    counts = _drop_counts([0.0, 0.0, 0.0], n_samples=6000)
    freq = counts / counts.sum()
    assert (freq - 1.0 / 3).abs().max().item() < 0.03


def test_identity_in_eval_mode():
    drop = ZeroEmbeddingDropout(p_apply=1.0).eval()
    x = torch.randn(8, 3, 4)
    out = drop(x, torch.zeros(1, 4), torch.tensor([0.9, 0.1, 0.1]))
    assert torch.equal(out, x)
