"""Regression tests for exact adjoints in IDFTLayer and ISTFTLayer."""

import pytest
import torch

from physioex.explain.posthoc.vidft import (
    DFTInputXGradient,
    DFTIntegratedGradients,
    DFTLayer,
    IDFTLayer,
)
from physioex.explain.posthoc.vistdft import (
    ISTFTLayer,
    STFTInputXGradient,
    STFTIntegratedGradients,
    STFTLayer,
)


class TestIDFTAdjoint:
    @pytest.mark.parametrize("n", [64, 65])
    @pytest.mark.parametrize("use_rfft", [True, False])
    def test_idft_gradcheck_real_valued(self, n, use_rfft):
        """IDFTLayer backward must match finite-difference Jacobian."""
        torch.manual_seed(0)
        dft = DFTLayer(dim=-1, use_rfft=use_rfft, real_valued=True)
        idft = IDFTLayer(n=n, dim=-1, use_rfft=use_rfft, real_valued=True)
        x = torch.randn(1, n, dtype=torch.float64)
        x_dft = dft(x).detach().requires_grad_(True)
        weights = torch.randn(1, n, dtype=torch.float64)

        def fn(z):
            out = idft(z)
            if out.is_complex():
                out = out.real
            return (out * weights).sum()

        assert torch.autograd.gradcheck(fn, (x_dft,), eps=1e-6, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("n", [64, 65])
    def test_dft_ig_and_ixg_completeness(self, n):
        """Real-valued DFT IntegratedGradients and InputXGradient must satisfy completeness."""
        torch.manual_seed(1)
        weights = torch.randn(n, dtype=torch.float64)

        def linear_f(x):
            return (x * weights).sum(dim=-1)

        x = torch.randn(2, n, dtype=torch.float64)
        expected = linear_f(x) - linear_f(torch.zeros_like(x))

        ig = DFTIntegratedGradients(
            f=linear_f, n=n, dim=-1, use_rfft=True, real_valued=True, steps=16
        )
        ixg = DFTInputXGradient(
            f=linear_f, n=n, dim=-1, use_rfft=True, real_valued=True
        )

        torch.testing.assert_close(ig(x).sum(dim=-1), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ixg(x).sum(dim=-1), expected, atol=1e-5, rtol=1e-5)

    def test_idft_complex_gradient_matches_native_irfft(self):
        """Complex-valued IDFTLayer gradient must match native torch.fft.irfft."""
        torch.manual_seed(2)
        n = 64
        x = torch.randn(2, n, dtype=torch.float64)
        w = torch.randn(2, n, dtype=torch.float64)
        x_dft = torch.fft.rfft(x, dim=-1)

        z_layer = x_dft.detach().requires_grad_(True)
        idft = IDFTLayer(n=n, dim=-1, use_rfft=True, real_valued=False)
        (idft(z_layer) * w).sum().backward()

        z_ref = x_dft.detach().requires_grad_(True)
        (torch.fft.irfft(z_ref, n=n, dim=-1) * w).sum().backward()

        torch.testing.assert_close(z_layer.grad, z_ref.grad, atol=1e-6, rtol=1e-6)


class TestISTFTAdjoint:
    def test_istft_gradcheck_real_valued(self):
        """ISTFTLayer backward must match finite-difference Jacobian."""
        torch.manual_seed(3)
        n_fft, hop, length = 16, 4, 48
        stft = STFTLayer(n_fft=n_fft, hop_length=hop, real_valued=True)
        istft = ISTFTLayer(n_fft=n_fft, length=length, hop_length=hop, real_valued=True)
        x = torch.randn(1, length, dtype=torch.float64)
        x_stft = stft(x).detach().requires_grad_(True)
        weights = torch.randn(1, length, dtype=torch.float64)

        def fn(z):
            return (istft(z) * weights).sum()

        assert torch.autograd.gradcheck(fn, (x_stft,), eps=1e-6, atol=1e-5, rtol=1e-4)

    def test_stft_ig_and_ixg_completeness(self):
        """Real-valued STFT IntegratedGradients and InputXGradient must satisfy completeness."""
        torch.manual_seed(4)
        n_fft, hop, length = 32, 8, 128
        weights = torch.randn(length, dtype=torch.float64)

        def linear_f(x):
            return (x * weights).sum(dim=-1)

        x = torch.randn(2, length, dtype=torch.float64)
        expected = linear_f(x) - linear_f(torch.zeros_like(x))

        ig = STFTIntegratedGradients(
            f=linear_f,
            n_fft=n_fft,
            length=length,
            hop_length=hop,
            real_valued=True,
            steps=16,
        )
        ixg = STFTInputXGradient(
            f=linear_f,
            n_fft=n_fft,
            length=length,
            hop_length=hop,
            real_valued=True,
        )

        torch.testing.assert_close(
            ig(x).sum(dim=(-2, -1)), expected, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            ixg(x).sum(dim=(-2, -1)), expected, atol=1e-4, rtol=1e-4
        )
