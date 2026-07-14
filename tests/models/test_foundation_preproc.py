"""Unit tests for physioex.models.foundation_preproc (pure-torch ops)."""
import pytest
import torch

from physioex.models import foundation_preproc as P


@pytest.fixture
def x():
    # (B=2, C=3, T=100) deterministic tensor
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 100, generator=g) * 50.0


# ── Normalization ────────────────────────────────────────────────────

def test_mean_center_removes_dc(x):
    out = P.mean_center(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-4)
    assert out.shape == x.shape


def test_clip_uv(x):
    out = P.clip_uv(x, max_uv=10.0)
    assert out.max() <= 10.0 + 1e-6
    assert out.min() >= -10.0 - 1e-6


def test_scale_div():
    x = torch.full((1, 1, 4), 100.0)
    assert torch.allclose(P.scale_div(x, 100.0), torch.ones(1, 1, 4))


def test_clip_and_scale_range(x):
    out = P.clip_and_scale(x, max_uv=100.0)
    assert out.max() <= 1.0 + 1e-6 and out.min() >= -1.0 - 1e-6


def test_zscore_unit_variance(x):
    out = P.zscore(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-4)
    assert torch.allclose(out.std(dim=-1), torch.ones(2, 3), atol=1e-2)


def test_zscore_clip_bounds(x):
    out = P.zscore_clip(x, sigma_clip=1.0)
    assert out.abs().max() <= 1.0 + 1e-6


def test_zscore_per_recording_shape_and_reject_bad_ndim(x):
    out = P.zscore_per_recording(x, sigma_clip=None)
    assert out.shape == x.shape
    # per-sample (over C,T) mean ~ 0
    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(2), atol=1e-4)
    with pytest.raises(ValueError):
        P.zscore_per_recording(torch.randn(3, 100))  # 2D not allowed


def test_q95_normalize(x):
    out = P.q95_normalize(x)
    assert out.shape == x.shape


def test_q95_normalize_with_stats():
    x = torch.ones(1, 2, 10)
    q95 = torch.tensor([2.0, 4.0])
    out = P.q95_normalize_with_stats(x, q95, eps=0.0)
    assert torch.allclose(out[0, 0], torch.full((10,), 0.5))
    assert torch.allclose(out[0, 1], torch.full((10,), 0.25))


def test_zscore_with_stats():
    x = torch.tensor([[[2.0, 4.0]]])  # (1,1,2)
    out = P.zscore_with_stats(x, torch.tensor([2.0]), torch.tensor([2.0]))
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0]))


def test_minmax_scale_to_pm1():
    x = torch.tensor([[[0.0, 5.0, 10.0]]])
    out = P.minmax_scale(x)
    assert torch.allclose(out[0, 0], torch.tensor([-1.0, 0.0, 1.0]))


# ── Channel operations ───────────────────────────────────────────────

def test_pad_channels_pads_and_truncates():
    x = torch.ones(2, 3, 10)
    padded = P.pad_channels(x, 5)
    assert padded.shape == (2, 5, 10)
    assert torch.all(padded[:, 3:, :] == 0)  # zero-filled tail

    truncated = P.pad_channels(x, 2)
    assert truncated.shape == (2, 2, 10)


def test_select_channels():
    x = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
    out = P.select_channels(x, [0, 2])
    assert out.shape == (2, 2, 4)
    assert torch.equal(out[:, 0, :], x[:, 0, :])
    assert torch.equal(out[:, 1, :], x[:, 2, :])


def test_strip_zero_channels():
    x = torch.ones(1, 3, 5)
    x[:, 1, :] = 0.0  # middle channel all-zero
    filtered, kept = P.strip_zero_channels(x)
    assert kept == [0, 2]
    assert filtered.shape == (1, 2, 5)


def test_strip_zero_channels_keeps_all_when_none_zero():
    x = torch.ones(1, 3, 5)
    filtered, kept = P.strip_zero_channels(x)
    assert kept == [0, 1, 2]
    assert filtered is x  # returns the same tensor unchanged


def test_map_channels_to_layout():
    x = torch.zeros(1, 2, 4)
    x[:, 0, :] = 1.0  # C3
    x[:, 1, :] = 2.0  # C4
    mapped, active = P.map_channels_to_layout(
        x, input_names=["C3", "C4"], target_layout=["C4", "CZ", "C3"]
    )
    assert mapped.shape == (1, 3, 4)
    # C4 -> index 0, C3 -> index 2; CZ (index 1) missing -> zeros
    assert torch.all(mapped[:, 0, :] == 2.0)
    assert torch.all(mapped[:, 1, :] == 0.0)
    assert torch.all(mapped[:, 2, :] == 1.0)
    assert sorted(active) == [0, 2]


def test_map_channels_to_layout_with_aliases():
    x = torch.ones(1, 1, 4) * 3.0
    mapped, active = P.map_channels_to_layout(
        x, input_names=["T3"], target_layout=["T7"], aliases={"T3": "T7"}
    )
    assert torch.all(mapped[:, 0, :] == 3.0)
    assert active == [0]


def test_create_padding_mask():
    mask = P.create_padding_mask(n_channels=4, n_valid=2, batch_size=3, device=torch.device("cpu"))
    assert mask.shape == (3, 4)
    assert mask.dtype == torch.bool
    assert torch.all(~mask[:, :2])   # first 2 real -> False
    assert torch.all(mask[:, 2:])    # rest padded -> True


# ── Recording stats ──────────────────────────────────────────────────

def test_compute_channel_stats():
    sig = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    stats = P.compute_channel_stats(sig)
    assert set(stats) == {"mean", "std", "q95", "min", "max"}
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["min"] == pytest.approx(-2.0)
    assert stats["max"] == pytest.approx(4.0)
