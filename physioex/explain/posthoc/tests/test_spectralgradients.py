"""Extensive unit tests for SpectralGradients (local IG formulation)."""

import pytest
import torch

from physioex.explain.posthoc.spectralgradients import SpectralGradients
from physioex.explain.posthoc.gradients import IntegratedGradients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic(x):
    return (x**2).sum()


def _batched_quadratic(x):
    return (x**2).sum(dim=-1, keepdim=True)


def _batched_multiclass(x):
    return torch.stack(
        [
            x.sum(dim=-1),
            (x**2).sum(dim=-1),
            (x**3).sum(dim=-1),
        ],
        dim=-1,
    )


class TinyCNN(torch.nn.Module):
    def __init__(self, n=100, n_classes=3):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 10, 5), torch.nn.ReLU(), torch.nn.Flatten()
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, n)).shape[-1]
        self.fc = torch.nn.Linear(flat, n_classes)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0)
        assert sg.fs == 100.0
        assert sg.freq_step == 1.0
        assert sg.steps == 10
        assert sg.path == "both"
        assert sg.target is None
        assert sg.expects_batch is False

    def test_custom_params(self):
        sg = SpectralGradients(
            f=_quadratic,
            fs=256.0,
            freq_step=4.0,
            steps=20,
            path="low_to_high",
            target=2,
        )
        assert sg.fs == 256.0
        assert sg.freq_step == 4.0
        assert sg.steps == 20
        assert sg.path == "low_to_high"
        assert sg.target == 2

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="path must be"):
            SpectralGradients(f=_quadratic, fs=100.0, path="diagonal")

    def test_invalid_fs_raises(self):
        with pytest.raises(ValueError, match="fs must be"):
            SpectralGradients(f=_quadratic, fs=0)

    def test_invalid_freq_step_raises(self):
        with pytest.raises(ValueError, match="freq_step must be"):
            SpectralGradients(f=_quadratic, fs=100.0, freq_step=0)

    def test_invalid_steps_raises(self):
        with pytest.raises(ValueError, match="steps must be"):
            SpectralGradients(f=_quadratic, fs=100.0, steps=1)


# ---------------------------------------------------------------------------
# Hz-to-bin conversion
# ---------------------------------------------------------------------------


class TestFreqConversion:
    def test_bin_step_exact(self):
        """freq_step=1 Hz with fs=100, n=100 → freq_res=1 Hz → 1 bin."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0)
        assert sg._bin_step(100) == 1

    def test_bin_step_multi(self):
        """freq_step=5 Hz with fs=100, n=100 → freq_res=1 Hz → 5 bins."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=5.0)
        assert sg._bin_step(100) == 5

    def test_bin_step_fractional(self):
        """freq_step=2 Hz with fs=100, n=200 → freq_res=0.5 Hz → 4 bins."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=2.0)
        assert sg._bin_step(200) == 4

    def test_bin_step_min_one(self):
        """freq_step smaller than freq_res still gives 1 bin."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=0.1)
        assert sg._bin_step(100) >= 1

    def test_n_bands(self):
        # fs=100, n=100 → n_freqs=51, freq_step=1 Hz → bin_step=1 → 51 bands
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0)
        assert sg.n_bands(100) == 51

        # freq_step=10 Hz → bin_step=10 → ceil(51/10) = 6 bands
        sg10 = SpectralGradients(f=_quadratic, fs=100.0, freq_step=10.0)
        assert sg10.n_bands(100) == 6

    def test_band_frequencies(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0)
        freqs = sg.band_frequencies(signal_length=100)
        assert freqs.shape[0] == 51
        assert freqs[0].item() == pytest.approx(0.0, abs=1e-5)
        assert freqs[-1].item() == pytest.approx(50.0, abs=1e-5)

    def test_band_frequencies_coarse(self):
        """10 Hz bands on fs=100, n=100."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=10.0)
        freqs = sg.band_frequencies(signal_length=100)
        assert freqs.shape[0] == 6
        # First band covers bins 0..9 → freqs 0,1,..,9 Hz → center 4.5
        assert freqs[0].item() == pytest.approx(4.5, abs=0.01)

    def test_band_frequencies_after_forward(self):
        """band_frequencies() without signal_length after a forward call."""
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=5.0, steps=3)
        x = torch.randn(1, 100)
        sg(x)
        freqs = sg.band_frequencies()
        assert freqs.shape[0] == sg.n_bands(100)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestShape:
    def test_1d_input(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0, steps=3)
        x = torch.randn(50)
        out = sg(x)
        assert out.shape == (1, sg.n_bands(50), 50)

    def test_2d_input(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0, steps=3)
        x = torch.randn(4, 50)
        out = sg(x)
        assert out.shape == (4, sg.n_bands(50), 50)

    def test_coarse_step_reduces_bands(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=10.0, steps=3)
        x = torch.randn(2, 100)
        out = sg(x)
        assert out.shape[1] == sg.n_bands(100)
        assert out.shape[1] < 51  # fewer than individual bins

    def test_all_paths_same_shape(self):
        x = torch.randn(2, 40)
        for path in ("both", "low_to_high", "high_to_low"):
            sg = SpectralGradients(f=_quadratic, fs=100.0, steps=3, path=path)
            out = sg(x)
            expected = (2, sg.n_bands(40), 40)
            assert out.shape == expected, f"path={path}: {out.shape}"


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_completeness_quadratic(self):
        torch.manual_seed(42)
        x = torch.randn(2, 50)

        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0, steps=50)
        sg_sum = sg(x).sum(dim=1)

        ig = IntegratedGradients(f=_quadratic, steps=300)
        ig_attr = ig(x)

        torch.testing.assert_close(sg_sum, ig_attr, atol=0.1, rtol=0.1)

    def test_completeness_scalar(self):
        torch.manual_seed(0)
        x = torch.randn(3, 40)

        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=1.0, steps=50)
        total = sg(x).sum(dim=(1, 2))
        expected = (x**2).sum(dim=-1)

        torch.testing.assert_close(total, expected, atol=0.2, rtol=0.2)

    def test_completeness_all_paths(self):
        torch.manual_seed(42)
        x = torch.randn(1, 30)
        f_diff = (x**2).sum().item()

        for path in ("low_to_high", "high_to_low", "both"):
            sg = SpectralGradients(f=_quadratic, fs=100.0, steps=50, path=path)
            total = sg(x).sum().item()
            assert (
                abs(total - f_diff) / abs(f_diff) < 0.15
            ), f"path={path}: {total:.4f} vs {f_diff:.4f}"

    def test_completeness_coarse_hz(self):
        """Completeness holds at any freq_step in Hz."""
        torch.manual_seed(42)
        x = torch.randn(1, 60)
        f_diff = (x**2).sum().item()

        for hz in [1.0, 5.0, 10.0, 25.0]:
            sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=hz, steps=50)
            total = sg(x).sum().item()
            assert (
                abs(total - f_diff) / abs(f_diff) < 0.15
            ), f"freq_step={hz} Hz: {total:.4f} vs {f_diff:.4f}"


# ---------------------------------------------------------------------------
# Path dependence
# ---------------------------------------------------------------------------


class TestPathDependence:
    def test_both_is_average(self):
        torch.manual_seed(42)
        x = torch.randn(1, 40)

        sg_fwd = SpectralGradients(f=_quadratic, fs=100.0, steps=20, path="low_to_high")
        sg_bwd = SpectralGradients(f=_quadratic, fs=100.0, steps=20, path="high_to_low")
        sg_both = SpectralGradients(f=_quadratic, fs=100.0, steps=20, path="both")

        expected = 0.5 * (sg_fwd(x) + sg_bwd(x))
        torch.testing.assert_close(sg_both(x), expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_finite(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, steps=5)
        assert torch.isfinite(sg(torch.randn(2, 60))).all()

    def test_zero_input(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, steps=5)
        out = sg(torch.zeros(1, 40))
        assert torch.isfinite(out).all()
        assert out.abs().max().item() < 1e-6

    def test_batch_consistency(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=5.0, steps=10)
        x = torch.randn(1, 40)
        out = sg(x.expand(3, -1).clone())
        torch.testing.assert_close(out[0], out[1], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Batched
# ---------------------------------------------------------------------------


class TestBatched:
    def test_matches_per_sample(self):
        torch.manual_seed(42)
        x = torch.randn(3, 40)

        sg_ps = SpectralGradients(
            f=_quadratic,
            fs=100.0,
            freq_step=5.0,
            steps=10,
            path="low_to_high",
        )
        sg_bt = SpectralGradients(
            f=_batched_quadratic,
            fs=100.0,
            freq_step=5.0,
            steps=10,
            path="low_to_high",
            expects_batch=True,
        )
        torch.testing.assert_close(sg_ps(x), sg_bt(x), atol=1e-4, rtol=1e-4)

    def test_batched_target(self):
        sg = SpectralGradients(
            f=_batched_multiclass,
            fs=100.0,
            expects_batch=True,
            target=1,
            freq_step=5.0,
            steps=5,
        )
        out = sg(torch.randn(2, 40))
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()

    def test_call_count_parallel(self):
        """Parallel path: f called once per IG step (all bands mega-batched)."""
        count = 0

        def f_counted(x):
            nonlocal count
            count += 1
            return (x**2).sum(dim=-1, keepdim=True)

        sg = SpectralGradients(
            f=f_counted,
            fs=100.0,
            freq_step=10.0,
            steps=5,
            path="low_to_high",
            expects_batch=True,
        )
        count = 0
        sg(torch.randn(4, 100))
        # Parallel: exactly `steps` calls, regardless of n_bands or B
        assert count == 5, f"expected 5, got {count}"

    def test_call_count_both_paths(self):
        """path='both' still uses only `steps` calls (both dirs mega-batched)."""
        count = 0

        def f_counted(x):
            nonlocal count
            count += 1
            return (x**2).sum(dim=-1, keepdim=True)

        sg = SpectralGradients(
            f=f_counted,
            fs=100.0,
            freq_step=10.0,
            steps=5,
            path="both",
            expects_batch=True,
        )
        count = 0
        sg(torch.randn(4, 100))
        assert count == 5, f"expected 5, got {count}"


# ---------------------------------------------------------------------------
# With a real model
# ---------------------------------------------------------------------------


class TestWithModel:
    def test_cnn_softmax(self):
        torch.manual_seed(0)
        model = TinyCNN(n=100).eval()
        f = lambda x: model(x).softmax(dim=-1)

        sg = SpectralGradients(
            f=f,
            fs=100.0,
            expects_batch=True,
            target=0,
            freq_step=5.0,
            steps=10,
        )
        out = sg(torch.randn(2, 100))

        assert out.shape == (2, sg.n_bands(100), 100)
        assert torch.isfinite(out).all()
        assert out.abs().max().item() > 1e-6

    def test_cnn_completeness(self):
        torch.manual_seed(0)
        model = TinyCNN(n=60).eval()

        x = torch.randn(1, 60)

        ig = IntegratedGradients(
            f=lambda x: model(x).softmax(-1).squeeze(0)[0],
            steps=200,
        )
        ig_attr = ig(x)

        sg = SpectralGradients(
            f=lambda x: model(x).softmax(-1),
            fs=100.0,
            expects_batch=True,
            target=0,
            freq_step=1.0,
            steps=20,
            path="both",
        )
        sg_sum = sg(x).sum(dim=1)

        rel_err = (sg_sum - ig_attr).abs() / ig_attr.abs().clamp(min=1e-8)
        assert rel_err.median().item() < 1.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_n_bands_matches_output(self):
        sg = SpectralGradients(f=_quadratic, fs=100.0, freq_step=5.0, steps=3)
        x = torch.randn(1, 50)
        assert sg(x).shape[1] == sg.n_bands(50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
