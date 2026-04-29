"""Extensive unit tests for explain.posthoc.vistdft.

Covers: STFTLayer, ISTFTLayer, _ISTFTObserveFn backward shape,
        STFTSaliency, STFTInputXGradient, STFTIntegratedGradients,
        STFTExpectedGradients.
"""

import pytest
import torch

from physioex.explain.posthoc.vistdft import (
    ISTFTLayer,
    STFTExpectedGradients,
    STFTInputXGradient,
    STFTIntegratedGradients,
    STFTLayer,
    STFTSaliency,
    _ISTFTObserveFn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic(x):
    return (x**2).sum()


def _linear(x):
    return x.sum()


def _multiclass(x):
    return torch.stack([x.sum(), (x**2).sum(), (x**3).sum()])


# Common STFT params
N_FFT = 64
HOP = 16
WIN = 64
SIGNAL_LEN = 256


def _stft_kwargs():
    return dict(n_fft=N_FFT, hop_length=HOP, win_length=WIN)


def _stft_explainer_kwargs():
    return dict(n_fft=N_FFT, length=SIGNAL_LEN, hop_length=HOP, win_length=WIN)


# ---------------------------------------------------------------------------
# STFTLayer
# ---------------------------------------------------------------------------


class TestSTFTLayer:
    def test_output_shape(self):
        layer = STFTLayer(**_stft_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        out = layer(x)
        assert out.ndim == 3
        assert out.shape[0] == 2
        # freq bins = n_fft // 2 + 1 (onesided)
        assert out.shape[1] == N_FFT // 2 + 1
        assert out.is_complex()

    def test_single_sample(self):
        layer = STFTLayer(**_stft_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        out = layer(x)
        assert out.shape[0] == 1

    def test_deterministic(self):
        layer = STFTLayer(**_stft_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# ISTFTLayer
# ---------------------------------------------------------------------------


class TestISTFTLayer:
    def test_output_shape(self):
        stft = STFTLayer(**_stft_kwargs())
        istft = ISTFTLayer(**_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        X = stft(x)
        y = istft(X)
        assert y.shape == (2, SIGNAL_LEN)

    def test_roundtrip(self):
        """STFT → ISTFT should approximately recover the signal."""
        stft = STFTLayer(**_stft_kwargs())
        istft = ISTFTLayer(**_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        x_rec = istft(stft(x))
        # Roundtrip is near-perfect with matching params
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# _ISTFTObserveFn backward count
# ---------------------------------------------------------------------------


class TestISTFTBackward:
    def test_backward_returns_correct_count(self):
        """backward must return exactly 9 values (matching 9 forward inputs after ctx)."""
        stft = STFTLayer(**_stft_kwargs())
        istft = ISTFTLayer(**_stft_explainer_kwargs())

        x = torch.randn(1, SIGNAL_LEN, requires_grad=True)
        X = stft(x)
        y = istft(X)
        loss = (y**2).sum()
        # This should not raise — if backward return count is wrong,
        # autograd will error here.
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_values_finite(self):
        stft = STFTLayer(**_stft_kwargs())
        istft = ISTFTLayer(**_stft_explainer_kwargs())

        x = torch.randn(3, SIGNAL_LEN, requires_grad=True)
        X = stft(x)
        y = istft(X)
        loss = y.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# STFTSaliency
# ---------------------------------------------------------------------------


class TestSTFTSaliency:
    def test_output_shape(self):
        sal = STFTSaliency(f=_quadratic, **_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        out = sal(x)
        # Output is in STFT domain: (batch, freq_bins, n_frames)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[1] == N_FFT // 2 + 1

    def test_finite_values(self):
        sal = STFTSaliency(f=_quadratic, **_stft_explainer_kwargs())
        x = torch.randn(3, SIGNAL_LEN)
        out = sal(x)
        assert torch.isfinite(out.abs()).all()

    def test_target_class(self):
        sal0 = STFTSaliency(f=_multiclass, target=0, **_stft_explainer_kwargs())
        sal1 = STFTSaliency(f=_multiclass, target=1, **_stft_explainer_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        out0 = sal0(x)
        out1 = sal1(x)
        assert not torch.allclose(out0.abs(), out1.abs(), atol=1e-3)

    def test_expects_batch(self):
        def f_batch(x):
            """Proper batched function: (B, D) -> (B, 1)."""
            return (x**2).sum(dim=-1, keepdim=True)

        sal = STFTSaliency(f=f_batch, expects_batch=True, **_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        out = sal(x)
        assert torch.isfinite(out.abs()).all()

    def test_batch_consistency(self):
        sal = STFTSaliency(f=_quadratic, **_stft_explainer_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        x_batch = x.expand(3, -1).clone()
        out = sal(x_batch)
        torch.testing.assert_close(out[0], out[1], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# STFTInputXGradient
# ---------------------------------------------------------------------------


class TestSTFTInputXGradient:
    def test_output_shape(self):
        ixg = STFTInputXGradient(f=_quadratic, **_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        out = ixg(x)
        assert out.ndim == 3
        assert out.shape[0] == 2

    def test_finite_values(self):
        ixg = STFTInputXGradient(f=_quadratic, **_stft_explainer_kwargs())
        x = torch.randn(3, SIGNAL_LEN)
        out = ixg(x)
        assert torch.isfinite(out.abs()).all()


# ---------------------------------------------------------------------------
# STFTIntegratedGradients
# ---------------------------------------------------------------------------


class TestSTFTIntegratedGradients:
    def test_output_shape(self):
        ig = STFTIntegratedGradients(f=_quadratic, steps=10, **_stft_explainer_kwargs())
        x = torch.randn(2, SIGNAL_LEN)
        out = ig(x)
        assert out.ndim == 3
        assert out.shape[0] == 2

    def test_finite_values(self):
        ig = STFTIntegratedGradients(f=_quadratic, steps=10, **_stft_explainer_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        out = ig(x)
        assert torch.isfinite(out.abs()).all()

    def test_baseline_passthrough(self):
        ig = STFTIntegratedGradients(f=_quadratic, steps=10, **_stft_explainer_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        attr_default = ig(x)
        baseline = torch.ones(1, SIGNAL_LEN)
        attr_custom = ig(x, baseline=baseline)
        assert not torch.allclose(attr_default.abs(), attr_custom.abs(), atol=1e-3)

    def test_steps_override(self):
        ig = STFTIntegratedGradients(f=_quadratic, steps=10, **_stft_explainer_kwargs())
        x = torch.randn(1, SIGNAL_LEN)
        out = ig(x, steps=5)
        assert out.ndim == 3


# ---------------------------------------------------------------------------
# STFTExpectedGradients
# ---------------------------------------------------------------------------


class TestSTFTExpectedGradients:
    def test_output_shape(self):
        baselines = torch.randn(10, SIGNAL_LEN)
        eg = STFTExpectedGradients(
            f=_quadratic, baselines=baselines, n_samples=10, **_stft_explainer_kwargs()
        )
        x = torch.randn(1, SIGNAL_LEN)
        out = eg(x)
        assert out.ndim == 3
        assert out.shape[0] == 1

    def test_finite_values(self):
        baselines = torch.randn(10, SIGNAL_LEN)
        eg = STFTExpectedGradients(
            f=_quadratic, baselines=baselines, n_samples=10, **_stft_explainer_kwargs()
        )
        x = torch.randn(2, SIGNAL_LEN)
        out = eg(x)
        assert torch.isfinite(out.abs()).all()

    def test_baselines_via_forward(self):
        baselines = torch.randn(10, SIGNAL_LEN)
        eg = STFTExpectedGradients(
            f=_quadratic, baselines=baselines, n_samples=10, **_stft_explainer_kwargs()
        )
        x = torch.randn(1, SIGNAL_LEN)
        new_baselines = torch.randn(15, SIGNAL_LEN)
        out = eg(x, baselines=new_baselines)
        assert torch.isfinite(out.abs()).all()


# ---------------------------------------------------------------------------
# Cross-cutting STFT tests
# ---------------------------------------------------------------------------


class TestSTFTCrossCutting:
    def test_all_methods_same_ndim(self):
        """All STFT methods produce 3D output."""
        x = torch.randn(2, SIGNAL_LEN)
        baselines = torch.randn(10, SIGNAL_LEN)
        kw = _stft_explainer_kwargs()

        explainers = [
            STFTSaliency(f=_quadratic, **kw),
            STFTInputXGradient(f=_quadratic, **kw),
            STFTIntegratedGradients(f=_quadratic, steps=5, **kw),
            STFTExpectedGradients(f=_quadratic, baselines=baselines, n_samples=5, **kw),
        ]

        for exp in explainers:
            out = exp(x)
            assert out.ndim == 3, f"{type(exp).__name__}: ndim={out.ndim}"
            assert out.shape[0] == 2

    def test_zero_input(self):
        x = torch.zeros(1, SIGNAL_LEN)
        sal = STFTSaliency(f=_quadratic, **_stft_explainer_kwargs())
        out = sal(x)
        assert torch.isfinite(out.abs()).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
