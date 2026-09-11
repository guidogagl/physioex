"""Shared LRP primitives (relevance-as-gradient convention).

All rules follow the convention used by Zennit and LXT: the *backward pass*
carries **relevance** instead of gradients, so a plain ``out.backward(seed)``
with ``seed = f_c(x)`` at the target neuron yields the LRP relevance in
``input.grad``.  Compared with LXT's functionals these implementations use a
**signed, dtype-preserving** stabiliser ``z + ε·sign(z)`` (Arras et al.;
Zennit's ``stabilize``) instead of the sign-blind ``z + ε`` — the sign-blind
form is unbounded for ``z ∈ (−2ε, 0)`` and promotes half-precision tensors to
float32.

Functions
---------
stabilize(z, eps)          signed stabiliser, ``z==0 → +eps``
linear_eps(x, W, b, eps)   ε-LRP for ``y = x Wᵀ + b`` (bias relevance absorbed)
add_eps(a, b, eps)         proportional split of ``y = a + b``
mul_signal_take(g, s)      all relevance to the *source* ``s`` (Arras)
st_identity(pre, act)      forward ``act``, backward identity to ``pre``
"""

from __future__ import annotations

import torch


def stabilize(z: torch.Tensor, eps: float) -> torch.Tensor:
    """``z + ε·sign(z)`` with ``sign(0) := +1``; keeps ``z``'s dtype/device."""
    sign = torch.sign(z)
    sign = sign + (sign == 0).to(z.dtype)
    return z + eps * sign


class _LinearEps(torch.autograd.Function):
    """ε-LRP through ``y = x Wᵀ + b``: ``R_x = x ⊙ ((R_y / (y ± ε)) W)``.

    Bias relevance is *absorbed* (not redistributed to the inputs), the standard
    LRP-ε convention; conservation therefore holds exactly only for ``b=None``.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y = torch.nn.functional.linear(x, weight, bias)
        ctx.save_for_backward(x, weight, y)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, relevance):
        x, weight, y = ctx.saved_tensors
        s = relevance / stabilize(y, ctx.eps)
        return x * torch.matmul(s, weight), None, None, None


def linear_eps(x, weight, bias=None, eps: float = 1e-6):
    return _LinearEps.apply(x, weight, bias, eps)


class _AddEps(torch.autograd.Function):
    """Proportional split of ``y = a + b``: ``R_a = a·R/(y±ε)``, ``R_b = b·R/(y±ε)``."""

    @staticmethod
    def forward(ctx, a, b, eps):
        y = a + b
        ctx.save_for_backward(a, b, y)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, relevance):
        a, b, y = ctx.saved_tensors
        n = relevance / stabilize(y, ctx.eps)
        r_a, r_b = n * a, n * b
        # un-broadcast to the operand shapes (residual adds are same-shape, but
        # be safe for e.g. (B,T,D)+(1,T,D) positional terms)
        if r_a.shape != a.shape:
            r_a = _sum_to_shape(r_a, a.shape)
        if r_b.shape != b.shape:
            r_b = _sum_to_shape(r_b, b.shape)
        return r_a, r_b, None


def _sum_to_shape(t, shape):
    while t.dim() > len(shape):
        t = t.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and t.shape[i] != 1:
            t = t.sum(i, keepdim=True)
    return t


def add_eps(a, b, eps: float = 1e-6):
    return _AddEps.apply(a, b, eps)


class _MulSignalTake(torch.autograd.Function):
    """``y = gate * source``; backward routes **all** relevance to ``source``."""

    @staticmethod
    def forward(ctx, gate, source):
        return gate * source

    @staticmethod
    def backward(ctx, relevance):
        return torch.zeros_like(relevance), relevance


def mul_signal_take(gate, source):
    return _MulSignalTake.apply(gate, source)


class _STIdentity(torch.autograd.Function):
    """Forward returns ``act`` bit-exactly; backward passes relevance to ``pre``."""

    @staticmethod
    def forward(ctx, pre, act):
        return act

    @staticmethod
    def backward(ctx, relevance):
        return relevance, None


def st_identity(pre, act):
    """Straight-through nonlinearity (identity rule): value ``act``, relevance → ``pre``."""
    return _STIdentity.apply(pre, act)


def target_seed(out: torch.Tensor, in_index: int, out_index: int):
    """Relevance seed selecting the target neuron with its **logit value**.

    Seeding with ``f_c(x)`` (not a bare ``1.0``) makes total relevance conserve
    to the class evidence, ``Σ R ≈ f_c(x)``.  Handles ``(B, L, n_classes)``
    sequence outputs and ``(B, n_classes)``.  Returns ``(seed, target_values)``.
    """
    seed = torch.zeros_like(out)
    if out.dim() == 3:
        target = out[:, in_index, out_index].detach()
        seed[:, in_index, out_index] = target
    elif out.dim() == 2:
        target = out[:, out_index].detach()
        seed[:, out_index] = target
    else:
        raise ValueError(f"Unexpected model output rank {out.dim()}; expected 2 or 3.")
    return seed, target
