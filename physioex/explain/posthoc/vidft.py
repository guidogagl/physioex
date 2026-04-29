import torch

import physioex.explain.posthoc.gradients as grads


class DFTLayer(torch.nn.Module):
    """Compute the (real) DFT of the input along *dim*."""

    def __init__(self, dim: int = -1, use_rfft: bool = True):
        super().__init__()
        self.dim = dim
        self.use_rfft = use_rfft

    def forward(self, x):
        if self.use_rfft:
            return torch.fft.rfft(x, dim=self.dim)
        return torch.fft.fft(x, dim=self.dim)


class _IDFTObserveFn(torch.autograd.Function):
    """Custom autograd for IDFT with explicit backward (STFT of grad)."""

    @staticmethod
    def forward(ctx, x, n, dim, use_rfft):
        ctx.n = n
        ctx.dim = dim
        ctx.use_rfft = use_rfft

        if use_rfft:
            return torch.fft.irfft(x, n=n, dim=dim)
        return torch.fft.ifft(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_rfft:
            grad_input = torch.fft.rfft(grad_output, dim=ctx.dim)
        else:
            grad_input = torch.fft.fft(grad_output, dim=ctx.dim)
        return grad_input, None, None, None


class IDFTLayer(torch.nn.Module):
    """Compute the inverse (real) DFT with a differentiable backward pass.

    Args:
        n: Output signal length.  **Must be > 0** to produce a valid signal.
        dim: Dimension along which to compute the IDFT.
        use_rfft: If True uses ``irfft`` / ``rfft`` pair.
    """

    def __init__(self, n: int, dim: int = -1, use_rfft: bool = True):
        super().__init__()
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self.n = n
        self.dim = dim
        self.use_rfft = use_rfft

    def forward(self, x):
        return _IDFTObserveFn.apply(x, self.n, self.dim, self.use_rfft)


# ---------------------------------------------------------------------------
# DFT-domain variants of the base gradient methods
# ---------------------------------------------------------------------------
# Each variant:
#   - transforms x to frequency domain in forward()
#   - overrides _scalar_f to convert back to time domain before calling f
#   - propagates *target* and other constructor args to the base class


def _dft_batched_scores(self, x_dft_batch):
    """Shared _batched_scores for all DFT* explainers (batched path).

    Converts DFT-domain batch to time domain, then evaluates f.
    """
    x_time = self.idft_layer(x_dft_batch)  # (B, n)
    out = self.f(x_time)  # (B, C) or (B,)
    B = x_dft_batch.shape[0]
    flat = out.view(B, -1)
    if self.target is not None:
        return flat[:, self.target]
    return flat[:, 0]


class DFTSaliency(grads.Saliency):
    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True, **kwargs
    ):
        super(DFTSaliency, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft)

    _batched_scores = _dft_batched_scores

    def _scalar_f(self, x):
        x = self.idft_layer(x)

        if self.expects_batch:
            out = self.f(x.unsqueeze(0))
        else:
            out = self.f(x)

        flat = out.view(-1)
        if self.target is not None:
            return flat[self.target]
        return flat[0]

    def forward(self, x):
        x_dft = self.dft_layer(x)
        return super(DFTSaliency, self).forward(x_dft)


class DFTInputXGradient(grads.InputXGradient):
    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True, **kwargs
    ):
        super(DFTInputXGradient, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft)

    _batched_scores = _dft_batched_scores

    def _scalar_f(self, x):
        x = self.idft_layer(x)

        if self.expects_batch:
            out = self.f(x.unsqueeze(0))
        else:
            out = self.f(x)

        flat = out.view(-1)
        if self.target is not None:
            return flat[self.target]
        return flat[0]

    def forward(self, x):
        x_dft = self.dft_layer(x)
        return super(DFTInputXGradient, self).forward(x_dft)


class DFTIntegratedGradients(grads.IntegratedGradients):
    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True, **kwargs
    ):
        super(DFTIntegratedGradients, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft)

    _batched_scores = _dft_batched_scores

    def _scalar_f(self, x):
        x = self.idft_layer(x)

        if self.expects_batch:
            out = self.f(x.unsqueeze(0))
        else:
            out = self.f(x)

        flat = out.view(-1)
        if self.target is not None:
            return flat[self.target]
        return flat[0]

    def forward(self, x, baseline=None, steps=None):
        x_dft = self.dft_layer(x)
        if baseline is not None:
            baseline_dft = self.dft_layer(baseline)
        else:
            baseline_dft = None
        return super(DFTIntegratedGradients, self).forward(
            x_dft, baseline=baseline_dft, steps=steps
        )


class DFTExpectedGradients(grads.ExpectedGradients):
    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True, **kwargs
    ):
        super(DFTExpectedGradients, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft)

    _batched_scores = _dft_batched_scores

    def _scalar_f(self, x):
        x = self.idft_layer(x)

        if self.expects_batch:
            out = self.f(x.unsqueeze(0))
        else:
            out = self.f(x)

        flat = out.view(-1)
        if self.target is not None:
            return flat[self.target]
        return flat[0]

    def forward(self, x, baselines=None, steps=None):
        x_dft = self.dft_layer(x)
        if baselines is not None:
            baselines_dft = self.dft_layer(baselines)
            self.set_baselines(baselines_dft)
        elif self.baselines is not None and self.baselines.shape[-1] != x_dft.shape[-1]:
            # Baselines were set in time domain at __init__; transform them now.
            baselines_dft = self.dft_layer(self.baselines)
            self.set_baselines(baselines_dft)
        return super(DFTExpectedGradients, self).forward(x_dft)
