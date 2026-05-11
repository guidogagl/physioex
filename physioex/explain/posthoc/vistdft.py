import torch

import physioex.explain.posthoc.gradients as grads


class STFTLayer(torch.nn.Module):
    """Compute the Short-Time Fourier Transform of the input.

    Args:
        real_valued: If True, returns ``[Re(X), Im(X)]`` concatenated
            along the frequency dimension (dim=-2) instead of a complex
            tensor.  Output shape becomes ``(B, 2*F, T)`` instead of
            ``(B, F, T)`` complex.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.real_valued = real_valued

        if window is None:
            window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, x):
        window = self.window.to(device=x.device, dtype=x.dtype)
        orig_shape = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1])
        X = torch.stft(
            x2d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        X = X.reshape(*orig_shape, X.shape[-2], X.shape[-1])
        if self.real_valued:
            return torch.cat([X.real, X.imag], dim=-2)
        return X


class _ISTFTObserveFn(torch.autograd.Function):
    """Custom autograd for ISTFT with explicit backward (STFT of grad).

    forward receives 10 inputs (after ctx):
        x, n_fft, hop_length, win_length, window, center, normalized, onesided, length, real_valued
    backward must return exactly 10 gradient values.

    When ``real_valued=True`` the input is a real tensor with
    ``[Re(X), Im(X)]`` concatenated along dim=-2.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        normalized,
        onesided,
        length,
        real_valued,
    ):
        ctx.n_fft = n_fft
        ctx.hop_length = hop_length
        ctx.win_length = win_length
        ctx.center = center
        ctx.normalized = normalized
        ctx.onesided = onesided
        ctx.length = length
        ctx.real_valued = real_valued

        if window is not None:
            window = window.to(device=x.device, dtype=x.real.dtype)
        ctx.save_for_backward(window)

        if real_valued:
            n_freq = x.shape[-2] // 2
            x_complex = torch.complex(x[..., :n_freq, :], x[..., n_freq:, :])
        else:
            x_complex = x

        return torch.istft(
            x_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
            length=length,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (window,) = ctx.saved_tensors
        if window is not None:
            window = window.to(device=grad_output.device, dtype=grad_output.real.dtype)

        grad2d = grad_output.reshape(-1, grad_output.shape[-1])
        G = torch.stft(
            grad2d,
            n_fft=ctx.n_fft,
            hop_length=ctx.hop_length,
            win_length=ctx.win_length,
            window=window,
            center=ctx.center,
            normalized=ctx.normalized,
            onesided=ctx.onesided,
            return_complex=True,
        )
        G = G.reshape(*grad_output.shape[:-1], G.shape[-2], G.shape[-1])
        if ctx.real_valued:
            grad_input = torch.cat([G.real, G.imag], dim=-2)
        else:
            grad_input = G
        # 10 inputs → 10 return values
        return grad_input, None, None, None, None, None, None, None, None, None


class ISTFTLayer(torch.nn.Module):
    """Compute the inverse STFT with a differentiable backward pass.

    Args:
        real_valued: If True, expects ``[Re(X), Im(X)]`` concatenated
            input along dim=-2 and produces real gradients in the same
            format.
    """

    def __init__(
        self,
        n_fft: int,
        length: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.length = length
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.real_valued = real_valued

        if window is None:
            window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, x):
        orig_shape = x.shape[:-2]
        x2d = x.reshape(-1, x.shape[-2], x.shape[-1])
        y2d = _ISTFTObserveFn.apply(
            x2d,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.normalized,
            self.onesided,
            self.length,
            self.real_valued,
        )
        return y2d.reshape(*orig_shape, self.length)


# ---------------------------------------------------------------------------
# STFT-domain variants of the base gradient methods
# ---------------------------------------------------------------------------


def _fold_real_valued_2d(attr):
    """Sum Re and Im relevances along freq dim: ``R_k = R_{k,Re} + R_{k,Im}``.

    Input has shape ``(..., 2*F, T)`` with Re in the first half and
    Im in the second half along dim=-2.  Returns ``(..., F, T)``.
    """
    n_freq = attr.shape[-2] // 2
    return attr[..., :n_freq, :] + attr[..., n_freq:, :]


def _stft_scalar_f(self, x):
    """Shared _scalar_f for all STFT* explainers (single-sample path)."""
    x = self.istft_layer(x)
    if self.expects_batch:
        out = self.f(x.unsqueeze(0))
    else:
        out = self.f(x)
    flat = out.view(-1)
    if self.target is not None:
        return flat[self.target]
    return flat[0]


def _stft_batched_scores(self, x_stft_batch):
    """Shared _batched_scores for all STFT* explainers (batched path).

    Converts STFT-domain batch to time domain, then evaluates f.
    """
    x_time = self.istft_layer(x_stft_batch)  # (B, length)
    out = self.f(x_time)  # (B, C) or (B,)
    B = x_stft_batch.shape[0]
    flat = out.view(B, -1)
    if self.target is not None:
        return flat[:, self.target]
    return flat[:, 0]


class STFTSaliency(grads.Saliency):
    """Saliency in the STFT domain.

    Args:
        real_valued: If True, uses the real-valued STFT formulation
            (Vielhaben et al. 2024) producing signed attributions.
    """

    def __init__(
        self,
        f: callable,
        n_fft: int,
        length: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
        **kwargs,
    ):
        super(STFTSaliency, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, normalized=normalized,
            onesided=onesided, real_valued=real_valued,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft, length=length, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            normalized=normalized, onesided=onesided, real_valued=real_valued,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x):
        x_stft = self.stft_layer(x)
        attr = super(STFTSaliency, self).forward(x_stft)
        if self.stft_layer.real_valued:
            attr = _fold_real_valued_2d(attr)
        return attr


class STFTInputXGradient(grads.InputXGradient):
    """Input x Gradient in the STFT domain.

    Args:
        real_valued: If True, uses the real-valued STFT formulation
            producing signed attributions.
    """

    def __init__(
        self,
        f: callable,
        n_fft: int,
        length: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
        **kwargs,
    ):
        super(STFTInputXGradient, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, normalized=normalized,
            onesided=onesided, real_valued=real_valued,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft, length=length, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            normalized=normalized, onesided=onesided, real_valued=real_valued,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x):
        x_stft = self.stft_layer(x)
        attr = super(STFTInputXGradient, self).forward(x_stft)
        if self.stft_layer.real_valued:
            attr = _fold_real_valued_2d(attr)
        return attr


class STFTIntegratedGradients(grads.IntegratedGradients):
    """Integrated Gradients in the STFT domain.

    Args:
        real_valued: If True, uses the real-valued STFT formulation
            producing signed attributions.
    """

    def __init__(
        self,
        f: callable,
        n_fft: int,
        length: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
        **kwargs,
    ):
        super(STFTIntegratedGradients, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, normalized=normalized,
            onesided=onesided, real_valued=real_valued,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft, length=length, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            normalized=normalized, onesided=onesided, real_valued=real_valued,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x, baseline=None, steps=None):
        x_stft = self.stft_layer(x)
        if baseline is not None:
            baseline_stft = self.stft_layer(baseline)
        else:
            baseline_stft = None
        attr = super(STFTIntegratedGradients, self).forward(
            x_stft, baseline=baseline_stft, steps=steps
        )
        if self.stft_layer.real_valued:
            attr = _fold_real_valued_2d(attr)
        return attr


class STFTExpectedGradients(grads.ExpectedGradients):
    """Expected Gradients in the STFT domain.

    Args:
        real_valued: If True, uses the real-valued STFT formulation
            producing signed attributions.
    """

    def __init__(
        self,
        f: callable,
        n_fft: int,
        length: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        real_valued: bool = False,
        **kwargs,
    ):
        super(STFTExpectedGradients, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, normalized=normalized,
            onesided=onesided, real_valued=real_valued,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft, length=length, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            normalized=normalized, onesided=onesided, real_valued=real_valued,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x, baselines=None, steps=None):
        x_stft = self.stft_layer(x)
        if baselines is not None:
            baselines_stft = self.stft_layer(baselines)
            self.set_baselines(baselines_stft)
        elif (
            self.baselines is not None and self.baselines.shape[1:] != x_stft.shape[1:]
        ):
            baselines_stft = self.stft_layer(self.baselines)
            self.set_baselines(baselines_stft)
        attr = super(STFTExpectedGradients, self).forward(x_stft)
        if self.stft_layer.real_valued:
            attr = _fold_real_valued_2d(attr)
        return attr
