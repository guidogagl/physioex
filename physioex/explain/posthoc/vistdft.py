import torch

import physioex.explain.posthoc.gradients as grads


class STFTLayer(torch.nn.Module):
    """Compute the Short-Time Fourier Transform of the input."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

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
        return X.reshape(*orig_shape, X.shape[-2], X.shape[-1])


class _ISTFTObserveFn(torch.autograd.Function):
    """Custom autograd for ISTFT with explicit backward (STFT of grad).

    forward receives 9 inputs (after ctx):
        x, n_fft, hop_length, win_length, window, center, normalized, onesided, length
    backward must return exactly 9 gradient values.
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
    ):
        ctx.n_fft = n_fft
        ctx.hop_length = hop_length
        ctx.win_length = win_length
        ctx.center = center
        ctx.normalized = normalized
        ctx.onesided = onesided
        ctx.length = length

        if window is not None:
            window = window.to(device=x.device, dtype=x.real.dtype)
        ctx.save_for_backward(window)

        return torch.istft(
            x,
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
        grad_input = G.reshape(*grad_output.shape[:-1], G.shape[-2], G.shape[-1])
        # 9 inputs → 9 return values: grad for x, then None for the 8 non-tensor args
        return grad_input, None, None, None, None, None, None, None, None


class ISTFTLayer(torch.nn.Module):
    """Compute the inverse STFT with a differentiable backward pass."""

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
    ):
        super().__init__()
        self.n_fft = n_fft
        self.length = length
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

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
        )
        return y2d.reshape(*orig_shape, self.length)


# ---------------------------------------------------------------------------
# STFT-domain variants of the base gradient methods
# ---------------------------------------------------------------------------


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
        **kwargs,
    ):
        super(STFTSaliency, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft,
            length=length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x):
        x_stft = self.stft_layer(x)
        return super(STFTSaliency, self).forward(x_stft)


class STFTInputXGradient(grads.InputXGradient):
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
        **kwargs,
    ):
        super(STFTInputXGradient, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft,
            length=length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x):
        x_stft = self.stft_layer(x)
        return super(STFTInputXGradient, self).forward(x_stft)


class STFTIntegratedGradients(grads.IntegratedGradients):
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
        **kwargs,
    ):
        super(STFTIntegratedGradients, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft,
            length=length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    _scalar_f = _stft_scalar_f
    _batched_scores = _stft_batched_scores

    def forward(self, x, baseline=None, steps=None):
        x_stft = self.stft_layer(x)
        if baseline is not None:
            baseline_stft = self.stft_layer(baseline)
        else:
            baseline_stft = None
        return super(STFTIntegratedGradients, self).forward(
            x_stft, baseline=baseline_stft, steps=steps
        )


class STFTExpectedGradients(grads.ExpectedGradients):
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
        **kwargs,
    ):
        super(STFTExpectedGradients, self).__init__(f, **kwargs)
        self.stft_layer = STFTLayer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.istft_layer = ISTFTLayer(
            n_fft=n_fft,
            length=length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
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
            # Baselines were set in time domain at __init__; transform them now.
            baselines_stft = self.stft_layer(self.baselines)
            self.set_baselines(baselines_stft)
        return super(STFTExpectedGradients, self).forward(x_stft)
