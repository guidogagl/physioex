import torch

import physioex.explain.posthoc.gradients as grads


class DFTLayer(torch.nn.Module):
    """Compute the (real) DFT of the input along *dim*.

    Args:
        dim: Dimension along which to compute the DFT.
        use_rfft: If True uses ``rfft`` (one-sided).
        real_valued: If True, returns ``[Re(X), Im(X)]`` concatenated
            along *dim* instead of a complex tensor.  This follows
            Vielhaben et al. (2024): the DFT is reformulated as a
            purely real linear layer so that gradient-based attributions
            are real and signed.
    """

    def __init__(self, dim: int = -1, use_rfft: bool = True, real_valued: bool = False):
        super().__init__()
        self.dim = dim
        self.use_rfft = use_rfft
        self.real_valued = real_valued

    def forward(self, x):
        if self.use_rfft:
            X = torch.fft.rfft(x, dim=self.dim)
        else:
            X = torch.fft.fft(x, dim=self.dim)
        if self.real_valued:
            return torch.cat([X.real, X.imag], dim=self.dim)
        return X


class _IDFTObserveFn(torch.autograd.Function):
    """Custom autograd for IDFT with explicit backward (DFT of grad).

    When ``real_valued=True`` the input is a real tensor of shape
    ``(..., 2*F)`` representing ``[Re(X), Im(X)]`` concatenated along
    *dim*.  The backward pass returns gradients in the same format.
    """

    @staticmethod
    def forward(ctx, x, n, dim, use_rfft, real_valued):
        ctx.n = n
        ctx.dim = dim
        ctx.use_rfft = use_rfft
        ctx.real_valued = real_valued

        if real_valued:
            n_freq = x.shape[dim] // 2
            re = x.narrow(dim, 0, n_freq)
            im = x.narrow(dim, n_freq, n_freq)
            x_complex = torch.complex(re, im)
        else:
            x_complex = x

        if use_rfft:
            return torch.fft.irfft(x_complex, n=n, dim=dim)
        return torch.fft.ifft(x_complex, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_rfft:
            G = torch.fft.rfft(grad_output, dim=ctx.dim)
        else:
            G = torch.fft.fft(grad_output, dim=ctx.dim)
        if ctx.real_valued:
            grad_input = torch.cat([G.real, G.imag], dim=ctx.dim)
        else:
            grad_input = G
        return grad_input, None, None, None, None


class IDFTLayer(torch.nn.Module):
    """Compute the inverse (real) DFT with a differentiable backward pass.

    Args:
        n: Output signal length.  **Must be > 0** to produce a valid signal.
        dim: Dimension along which to compute the IDFT.
        use_rfft: If True uses ``irfft`` / ``rfft`` pair.
        real_valued: If True, expects ``[Re(X), Im(X)]`` concatenated
            input and produces real gradients in the same format.
    """

    def __init__(self, n: int, dim: int = -1, use_rfft: bool = True, real_valued: bool = False):
        super().__init__()
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self.n = n
        self.dim = dim
        self.use_rfft = use_rfft
        self.real_valued = real_valued

    def forward(self, x):
        return _IDFTObserveFn.apply(x, self.n, self.dim, self.use_rfft, self.real_valued)


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


def _fold_real_valued(attr, dim=-1):
    """Sum Re and Im relevances: ``R_k = R_{k,Re} + R_{k,Im}``.

    Input has shape ``(..., 2*F, ...)`` with Re in the first half and
    Im in the second half along *dim*.  Returns ``(..., F, ...)``.
    """
    n_freq = attr.shape[dim] // 2
    re = attr.narrow(dim, 0, n_freq)
    im = attr.narrow(dim, n_freq, n_freq)
    return re + im


class DFTSaliency(grads.Saliency):
    """Saliency in the DFT domain.

    Args:
        real_valued: If True, uses the real-valued DFT formulation
            (Vielhaben et al. 2024) producing signed attributions.
    """

    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True,
        real_valued: bool = False, **kwargs
    ):
        super(DFTSaliency, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft, real_valued=real_valued)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft, real_valued=real_valued)

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
        attr = super(DFTSaliency, self).forward(x_dft)
        if self.dft_layer.real_valued:
            attr = _fold_real_valued(attr, dim=self.dim)
        return attr


class DFTInputXGradient(grads.InputXGradient):
    """Input x Gradient in the DFT domain.

    Args:
        real_valued: If True, uses the real-valued DFT formulation
            producing signed attributions.
    """

    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True,
        real_valued: bool = False, **kwargs
    ):
        super(DFTInputXGradient, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft, real_valued=real_valued)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft, real_valued=real_valued)

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
        attr = super(DFTInputXGradient, self).forward(x_dft)
        if self.dft_layer.real_valued:
            attr = _fold_real_valued(attr, dim=self.dim)
        return attr


class DFTIntegratedGradients(grads.IntegratedGradients):
    """Integrated Gradients in the DFT domain.

    Args:
        real_valued: If True, uses the real-valued DFT formulation
            producing signed attributions.
    """

    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True,
        real_valued: bool = False, **kwargs
    ):
        super(DFTIntegratedGradients, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft, real_valued=real_valued)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft, real_valued=real_valued)

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
        attr = super(DFTIntegratedGradients, self).forward(
            x_dft, baseline=baseline_dft, steps=steps
        )
        if self.dft_layer.real_valued:
            attr = _fold_real_valued(attr, dim=self.dim)
        return attr


class DFTExpectedGradients(grads.ExpectedGradients):
    """Expected Gradients in the DFT domain.

    Args:
        real_valued: If True, uses the real-valued DFT formulation
            producing signed attributions.
    """

    def __init__(
        self, f: callable, n: int = None, dim: int = -1, use_rfft: bool = True,
        real_valued: bool = False, **kwargs
    ):
        super(DFTExpectedGradients, self).__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft

        if n is None or n <= 0:
            raise ValueError(f"n (signal length) must be > 0, got {n}")

        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft, real_valued=real_valued)
        self.idft_layer = IDFTLayer(n=n, dim=dim, use_rfft=use_rfft, real_valued=real_valued)

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
            baselines_dft = self.dft_layer(self.baselines)
            self.set_baselines(baselines_dft)
        attr = super(DFTExpectedGradients, self).forward(x_dft)
        if self.dft_layer.real_valued:
            attr = _fold_real_valued(attr, dim=self.dim)
        return attr
