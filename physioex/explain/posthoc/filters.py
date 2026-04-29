import torch


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else ndim + dim


def _prod_or_one(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return torch.prod(x)


def _poly_from_roots(roots: torch.Tensor) -> torch.Tensor:
    coeffs = torch.ones(1, dtype=roots.dtype, device=roots.device)
    for r in roots:
        coeffs = torch.cat(
            [coeffs, torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)]
        )
        coeffs[1:] = coeffs[1:] - r * coeffs[:-1].clone()
    return coeffs


def _zpk2tf(
    z: torch.Tensor, p: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b = _poly_from_roots(z) * k
    a = _poly_from_roots(p)
    if torch.is_complex(b):
        if torch.max(torch.abs(b.imag)) < 1e-12:
            b = b.real
    if torch.is_complex(a):
        if torch.max(torch.abs(a.imag)) < 1e-12:
            a = a.real
    return b, a


def _butterworth_response(
    freqs: torch.Tensor, cutoff: float, order: int, btype: str
) -> torch.Tensor:
    if btype == "lowpass":
        ratio = freqs / cutoff
    elif btype == "highpass":
        eps = torch.finfo(freqs.dtype).eps
        ratio = cutoff / freqs.clamp_min(eps)
    else:
        raise ValueError("btype must be 'lowpass' or 'highpass'")

    response = 1.0 / torch.sqrt(1.0 + ratio.pow(2 * order))
    if btype == "highpass":
        response = torch.where(freqs == 0, torch.zeros_like(response), response)
    return response


def _fft_filter(
    x: torch.Tensor,
    cutoff: float,
    fs: float,
    order: int,
    btype: str,
    dim: int,
) -> torch.Tensor:
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError("cutoff must be in (0, fs/2)")
    if order < 1:
        raise ValueError("order must be >= 1")

    dim = _normalize_dim(dim, x.ndim)
    n = x.size(dim)
    dtype = x.real.dtype

    if x.is_complex():
        freqs = torch.fft.fftfreq(n, d=1.0 / fs, device=x.device, dtype=dtype).abs()
        X = torch.fft.fft(x, dim=dim)
    else:
        freqs = torch.fft.rfftfreq(n, d=1.0 / fs, device=x.device, dtype=dtype)
        X = torch.fft.rfft(x, dim=dim)

    response = _butterworth_response(freqs, cutoff, order, btype)
    response = response * response

    shape = [1] * X.ndim
    shape[dim] = response.numel()
    response = response.view(shape)

    Y = X * response
    if x.is_complex():
        return torch.fft.ifft(Y, dim=dim)
    return torch.fft.irfft(Y, n=n, dim=dim)


def _buttap(
    order: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype_c = (
        torch.complex128
        if dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    k = torch.arange(order, device=device, dtype=dtype)
    poles = torch.exp(1j * torch.pi * (2 * k + 1 + order) / (2 * order)).to(dtype_c)
    z = torch.empty(0, device=device, dtype=dtype_c)
    k = torch.ones((), device=device, dtype=dtype_c)
    return z, poles, k


def _lp2lp_zpk(
    z: torch.Tensor, p: torch.Tensor, k: torch.Tensor, wo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    degree = p.numel() - z.numel()
    z_lp = z * wo
    p_lp = p * wo
    k_lp = k * (wo**degree)
    return z_lp, p_lp, k_lp


def _lp2hp_zpk(
    z: torch.Tensor, p: torch.Tensor, k: torch.Tensor, wo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    degree = p.numel() - z.numel()
    z_hp = torch.empty(0, device=z.device, dtype=z.dtype) if z.numel() == 0 else wo / z
    p_hp = wo / p
    zeros_at_origin = torch.zeros(degree, device=z.device, dtype=z.dtype)
    z_hp = torch.cat([z_hp, zeros_at_origin])
    k_hp = k * torch.real(_prod_or_one(-z) / _prod_or_one(-p))
    return z_hp, p_hp, k_hp


def _bilinear_zpk(
    z: torch.Tensor, p: torch.Tensor, k: torch.Tensor, fs: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    degree = p.numel() - z.numel()
    two_fs = torch.tensor(2.0 * fs, device=z.device, dtype=z.dtype)
    z_d = (two_fs + z) / (two_fs - z)
    p_d = (two_fs + p) / (two_fs - p)
    if degree > 0:
        z_d = torch.cat([z_d, -torch.ones(degree, device=z.device, dtype=z.dtype)])
    k_d = k * torch.real(_prod_or_one(two_fs - z) / _prod_or_one(two_fs - p))
    k_d = k_d * (two_fs**degree)
    return z_d, p_d, k_d


def _butter_ba(
    order: int,
    cutoff: float,
    fs: float,
    btype: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError("cutoff must be in (0, fs/2)")
    wn = cutoff / (fs / 2)
    warped = (
        2.0
        * fs
        * torch.tan(torch.pi * torch.tensor(wn, device=device, dtype=dtype) / 2.0)
    )

    z, p, k = _buttap(order, device, dtype)
    if btype == "lowpass":
        z, p, k = _lp2lp_zpk(z, p, k, warped)
    elif btype == "highpass":
        z, p, k = _lp2hp_zpk(z, p, k, warped)
    else:
        raise ValueError("btype must be 'lowpass' or 'highpass'")

    z, p, k = _bilinear_zpk(z, p, k, fs)
    b, a = _zpk2tf(z, p, k)
    b = b / a[0]
    a = a / a[0]
    return b, a


def _lfilter_zi(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    n = max(a.numel(), b.numel())
    if n == 1:
        return torch.zeros(0, device=b.device, dtype=b.dtype)

    if a[0] != 1:
        b = b / a[0]
        a = a / a[0]

    b = torch.nn.functional.pad(b, (0, n - b.numel()))
    a = torch.nn.functional.pad(a, (0, n - a.numel()))
    a1 = a[1:]

    A = torch.zeros((n - 1, n - 1), device=b.device, dtype=b.dtype)
    A[0, :] = -a1
    if n > 2:
        A[1:, :-1] = torch.eye(n - 2, device=b.device, dtype=b.dtype)

    B = b[1:] - b[0] * a1
    I = torch.eye(n - 1, device=b.device, dtype=b.dtype)
    zi = torch.linalg.solve(I - A, B)
    return zi


def _lfilter_lastdim(
    b: torch.Tensor, a: torch.Tensor, x: torch.Tensor, zi: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(a.numel(), b.numel())
    if a[0] != 1:
        b = b / a[0]
        a = a / a[0]

    b = torch.nn.functional.pad(b, (0, n - b.numel()))
    a = torch.nn.functional.pad(a, (0, n - a.numel()))

    batch_shape = x.shape[:-1]
    t = x.shape[-1]
    x2 = x.reshape(-1, t)

    if n == 1:
        y = b[0] * x2
        return y.reshape(*batch_shape, t), torch.zeros(
            (x2.shape[0], 0), device=x.device, dtype=x.dtype
        )

    if zi is None:
        zi = torch.zeros((x2.shape[0], n - 1), device=x.device, dtype=x.dtype)

    y = torch.empty_like(x2)
    for i in range(t):
        x_i = x2[:, i]
        y_i = b[0] * x_i + zi[:, 0]
        y[:, i] = y_i

        new_zi = torch.empty_like(zi)
        if n > 2:
            new_zi[:, :-1] = (
                zi[:, 1:] + b[1:-1] * x_i.unsqueeze(-1) - a[1:-1] * y_i.unsqueeze(-1)
            )
        new_zi[:, -1] = b[-1] * x_i - a[-1] * y_i
        zi = new_zi

    return y.reshape(*batch_shape, t), zi


def _pad_signal(x: torch.Tensor, padlen: int, padtype: str) -> torch.Tensor:
    if padlen == 0:
        return x
    if padtype not in ("odd", "even", "constant"):
        raise ValueError("padtype must be 'odd', 'even', or 'constant'")

    left = x[..., 1 : padlen + 1].flip(-1)
    right = x[..., -padlen - 1 : -1].flip(-1)
    x0 = x[..., :1]
    xN = x[..., -1:]

    if padtype == "odd":
        left = 2 * x0 - left
        right = 2 * xN - right
    elif padtype == "constant":
        left = x0.expand(*x.shape[:-1], padlen)
        right = xN.expand(*x.shape[:-1], padlen)

    return torch.cat([left, x, right], dim=-1)


def filtfilt(
    b: torch.Tensor,
    a: torch.Tensor,
    x: torch.Tensor,
    dim: int = -1,
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> torch.Tensor:
    dim = _normalize_dim(dim, x.ndim)
    x = torch.movedim(x, dim, -1)
    n = x.shape[-1]

    if padtype is None:
        padlen = 0
    if padlen is None:
        padlen = 3 * (max(a.numel(), b.numel()) - 1)
    if padlen > 0 and n <= padlen:
        raise ValueError("input is too short for the required padding")

    if padlen > 0:
        x = _pad_signal(x, padlen, padtype)

    zi = _lfilter_zi(b, a).to(device=x.device, dtype=x.dtype)
    if zi.numel() > 0:
        zi_f = zi.view(1, -1).expand(x.reshape(-1, x.shape[-1]).shape[0], -1)
        zi_f = zi_f * x.reshape(-1, x.shape[-1])[:, 0].unsqueeze(-1)
    else:
        zi_f = None

    y, _ = _lfilter_lastdim(b, a, x, zi=zi_f)
    y = torch.flip(y, dims=[-1])

    if zi.numel() > 0:
        zi_b = zi.view(1, -1).expand(y.reshape(-1, y.shape[-1]).shape[0], -1)
        zi_b = zi_b * y.reshape(-1, y.shape[-1])[:, 0].unsqueeze(-1)
    else:
        zi_b = None

    y, _ = _lfilter_lastdim(b, a, y, zi=zi_b)
    y = torch.flip(y, dims=[-1])

    if padlen > 0:
        y = y[..., padlen:-padlen]

    return torch.movedim(y, -1, dim)


def lowpass_filter(
    x: torch.Tensor,
    cutoff: float,
    fs: float,
    order: int = 5,
    dim: int = -1,
) -> torch.Tensor:
    return _fft_filter(x, cutoff, fs, order, "lowpass", dim)


def highpass_filter(
    x: torch.Tensor,
    cutoff: float,
    fs: float,
    order: int = 5,
    dim: int = -1,
) -> torch.Tensor:
    return _fft_filter(x, cutoff, fs, order, "highpass", dim)
