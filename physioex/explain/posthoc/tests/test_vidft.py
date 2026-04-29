"""Extensive unit tests for explain.posthoc.vidft.

Covers: DFTLayer, IDFTLayer, DFTSaliency, DFTInputXGradient,
        DFTIntegratedGradients, DFTExpectedGradients.
"""

import pytest
import torch

from physioex.explain.posthoc.vidft import (
    DFTExpectedGradients,
    DFTInputXGradient,
    DFTIntegratedGradients,
    DFTLayer,
    DFTSaliency,
    IDFTLayer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic(x):
    """f(x) = sum(x^2)."""
    return (x**2).sum()


def _linear(x):
    """f(x) = sum(x)."""
    return x.sum()


def _multiclass(x):
    """Returns [sum(x), sum(x^2), sum(x^3)]."""
    return torch.stack([x.sum(), (x**2).sum(), (x**3).sum()])


# ---------------------------------------------------------------------------
# DFTLayer / IDFTLayer
# ---------------------------------------------------------------------------


class TestDFTLayer:
    def test_rfft_output_shape(self):
        layer = DFTLayer(dim=-1, use_rfft=True)
        x = torch.randn(2, 100)
        out = layer(x)
        assert out.shape == (2, 51)  # rfft: n//2+1
        assert out.is_complex()

    def test_fft_output_shape(self):
        layer = DFTLayer(dim=-1, use_rfft=False)
        x = torch.randn(2, 100)
        out = layer(x)
        assert out.shape == (2, 100)
        assert out.is_complex()

    def test_dim_parameter(self):
        layer = DFTLayer(dim=0, use_rfft=True)
        x = torch.randn(64, 3)
        out = layer(x)
        assert out.shape[0] == 33  # rfft along dim 0: 64//2+1

    def test_deterministic(self):
        layer = DFTLayer()
        x = torch.randn(1, 50)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)


class TestIDFTLayer:
    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            IDFTLayer(n=0)
        with pytest.raises(ValueError, match="n must be > 0"):
            IDFTLayer(n=-1)

    def test_roundtrip_rfft(self):
        """DFT → IDFT should recover original signal."""
        n = 100
        dft = DFTLayer(dim=-1, use_rfft=True)
        idft = IDFTLayer(n=n, dim=-1, use_rfft=True)
        x = torch.randn(3, n)
        x_rec = idft(dft(x))
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_roundtrip_fft(self):
        n = 64
        dft = DFTLayer(dim=-1, use_rfft=False)
        idft = IDFTLayer(n=n, dim=-1, use_rfft=False)
        x = torch.randn(2, n)
        x_rec = idft(dft(x))
        # fft/ifft may return complex; take real part
        if x_rec.is_complex():
            x_rec = x_rec.real
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        """Gradients should propagate through IDFTLayer."""
        n = 50
        dft = DFTLayer(dim=-1, use_rfft=True)
        idft = IDFTLayer(n=n, dim=-1, use_rfft=True)
        x = torch.randn(1, n, requires_grad=True)
        x_dft = dft(x)
        x_rec = idft(x_dft)
        loss = (x_rec**2).sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_output_shape(self):
        idft = IDFTLayer(n=100, dim=-1, use_rfft=True)
        x_dft = torch.randn(2, 51, dtype=torch.cfloat)
        out = idft(x_dft)
        assert out.shape == (2, 100)
        assert not out.is_complex()


# ---------------------------------------------------------------------------
# DFTSaliency
# ---------------------------------------------------------------------------


class TestDFTSaliency:
    def test_n_required(self):
        with pytest.raises(ValueError, match="n.*must be > 0"):
            DFTSaliency(f=_quadratic, n=None)
        with pytest.raises(ValueError, match="n.*must be > 0"):
            DFTSaliency(f=_quadratic, n=0)

    def test_output_shape(self):
        n = 64
        sal = DFTSaliency(f=_quadratic, n=n)
        x = torch.randn(2, n)
        out = sal(x)
        # Output is in frequency domain: (batch, n//2+1)
        assert out.shape == (2, n // 2 + 1)

    def test_output_is_complex(self):
        """DFT-domain attributions are complex-valued."""
        n = 32
        sal = DFTSaliency(f=_quadratic, n=n)
        x = torch.randn(1, n)
        out = sal(x)
        assert out.is_complex()

    def test_finite_values(self):
        n = 48
        sal = DFTSaliency(f=_quadratic, n=n)
        x = torch.randn(3, n)
        out = sal(x)
        assert torch.isfinite(out.abs()).all()

    def test_target_class(self):
        n = 32
        sal0 = DFTSaliency(f=_multiclass, n=n, target=0)
        sal1 = DFTSaliency(f=_multiclass, n=n, target=1)
        x = torch.randn(1, n)
        out0 = sal0(x)
        out1 = sal1(x)
        # Different targets should generally give different attributions
        assert not torch.allclose(out0.abs(), out1.abs(), atol=1e-3)

    def test_batch_consistency(self):
        """Same input in batch should give same attributions."""
        n = 40
        sal = DFTSaliency(f=_quadratic, n=n)
        x = torch.randn(1, n)
        x_batch = x.expand(3, -1).clone()
        out = sal(x_batch)
        torch.testing.assert_close(out[0], out[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[1], out[2], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# DFTInputXGradient
# ---------------------------------------------------------------------------


class TestDFTInputXGradient:
    def test_output_shape(self):
        n = 64
        ixg = DFTInputXGradient(f=_quadratic, n=n)
        x = torch.randn(2, n)
        out = ixg(x)
        assert out.shape == (2, n // 2 + 1)

    def test_n_required(self):
        with pytest.raises(ValueError):
            DFTInputXGradient(f=_quadratic, n=0)

    def test_finite_values(self):
        n = 32
        ixg = DFTInputXGradient(f=_quadratic, n=n)
        x = torch.randn(4, n)
        out = ixg(x)
        assert torch.isfinite(out.abs()).all()


# ---------------------------------------------------------------------------
# DFTIntegratedGradients
# ---------------------------------------------------------------------------


class TestDFTIntegratedGradients:
    def test_output_shape(self):
        n = 64
        ig = DFTIntegratedGradients(f=_quadratic, n=n, steps=10)
        x = torch.randn(2, n)
        out = ig(x)
        assert out.shape == (2, n // 2 + 1)

    def test_n_required(self):
        with pytest.raises(ValueError):
            DFTIntegratedGradients(f=_quadratic, n=0)

    def test_finite_values(self):
        n = 48
        ig = DFTIntegratedGradients(f=_quadratic, n=n, steps=20)
        x = torch.randn(3, n)
        out = ig(x)
        assert torch.isfinite(out.abs()).all()

    def test_baseline_passthrough(self):
        """Custom baseline should produce different results than default."""
        n = 32
        ig = DFTIntegratedGradients(f=_quadratic, n=n, steps=20)
        x = torch.randn(1, n)
        attr_default = ig(x)
        baseline = torch.ones(1, n)
        attr_custom = ig(x, baseline=baseline)
        # Results should differ when baseline changes
        assert not torch.allclose(attr_default.abs(), attr_custom.abs(), atol=1e-3)

    def test_steps_override(self):
        n = 32
        ig = DFTIntegratedGradients(f=_quadratic, n=n, steps=10)
        x = torch.randn(1, n)
        out = ig(x, steps=5)
        assert out.shape == (1, n // 2 + 1)


# ---------------------------------------------------------------------------
# DFTExpectedGradients
# ---------------------------------------------------------------------------


class TestDFTExpectedGradients:
    def test_output_shape(self):
        n = 32
        baselines = torch.randn(20, n)
        eg = DFTExpectedGradients(f=_quadratic, n=n, baselines=baselines, n_samples=10)
        x = torch.randn(2, n)
        out = eg(x)
        assert out.shape == (2, n // 2 + 1)

    def test_n_required(self):
        with pytest.raises(ValueError):
            DFTExpectedGradients(f=_quadratic, n=0, baselines=torch.randn(5, 10))

    def test_baselines_via_forward(self):
        """Passing baselines in forward should update and work."""
        n = 32
        initial_baselines = torch.randn(10, n)
        eg = DFTExpectedGradients(
            f=_quadratic, n=n, baselines=initial_baselines, n_samples=10
        )
        x = torch.randn(1, n)
        new_baselines = torch.randn(15, n)
        # This should work without error
        out = eg(x, baselines=new_baselines)
        assert out.shape == (1, n // 2 + 1)
        assert torch.isfinite(out.abs()).all()

    def test_no_state_mutation_across_calls(self):
        """Second call without baselines should still work with previously set baselines."""
        n = 32
        baselines = torch.randn(10, n)
        eg = DFTExpectedGradients(f=_quadratic, n=n, baselines=baselines, n_samples=10)
        x = torch.randn(1, n)
        out1 = eg(x)
        out2 = eg(x)
        # Both should succeed (baselines still available)
        assert out1.shape == out2.shape


# ---------------------------------------------------------------------------
# Cross-cutting DFT tests
# ---------------------------------------------------------------------------


class TestDFTCrossCutting:
    def test_all_dft_methods_same_freq_shape(self):
        """All DFT methods produce (batch, n//2+1) shaped output."""
        n = 40
        x = torch.randn(2, n)
        baselines = torch.randn(10, n)
        expected_shape = (2, n // 2 + 1)

        sal = DFTSaliency(f=_quadratic, n=n)
        ixg = DFTInputXGradient(f=_quadratic, n=n)
        ig = DFTIntegratedGradients(f=_quadratic, n=n, steps=10)
        eg = DFTExpectedGradients(f=_quadratic, n=n, baselines=baselines, n_samples=10)

        for name, explainer in [("sal", sal), ("ixg", ixg), ("ig", ig), ("eg", eg)]:
            out = explainer(x)
            assert (
                out.shape == expected_shape
            ), f"DFT{name}: {out.shape} != {expected_shape}"

    def test_zero_input(self):
        n = 32
        x = torch.zeros(1, n)
        sal = DFTSaliency(f=_quadratic, n=n)
        out = sal(x)
        assert torch.isfinite(out.abs()).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
