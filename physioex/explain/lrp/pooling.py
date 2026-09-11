"""CP-LRP for softmax attention-pooling blocks.

PhysioEx's custom attention poolings all collapse a sequence with softmax
weights: ``out = Σ_t softmax(score(x))_t · x_t`` — ``AttentionPooling``
(sleeptransformer), ``AttentionLayer`` (seqsleepnet, reused by lseqsleepnet and
protosleepnet), and ``ChannelMixer.attn_pool`` (protosleepnet).  Their softmax
leaks relevance just like attention, so the conservative-propagation (CP-LRP)
rule applies: treat the pooling weights as **constant** and route relevance
through the value path, which conserves exactly.

Requires the ``explain`` extra.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CPWeightedPool(torch.autograd.Function):
    """``out = Σ_t w_t · x_t`` over the sequence dim, weights constant.

    Backward (ε-LRP through the linear pooling with fixed ``w``):
    ``R(x[t,d]) = x[t,d] · w[t] · R_out[d] / (out[d]+ε)``.
    """

    @staticmethod
    def forward(ctx, x, weights, epsilon):
        out = (x * weights).sum(dim=1)  # (B, D)
        ctx.save_for_backward(x, weights, out)
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, relevance):
        x, weights, out = ctx.saved_tensors
        eps = ctx.epsilon
        denom = out + torch.where(out >= 0, eps, -eps)
        s = (relevance / denom).unsqueeze(1)  # (B, 1, D)
        r_x = x * weights * s
        return r_x, None, None


def cp_weighted_pool(x, weights, epsilon=1e-6):
    """Weighted sum over ``dim=1`` with the CP-LRP backward.  ``weights`` is
    ``(B, T, 1)`` and must be detached (constant) by the caller."""
    return _CPWeightedPool.apply(x, weights, epsilon)


class LRPAttentionPooling(nn.Module):
    """CP-LRP for ``sleeptransformer.AttentionPooling`` (and any pooling with a
    ``self.attention`` MLP + softmax over ``dim=1``)."""

    def __init__(self, orig: nn.Module, epsilon: float = 1e-6):
        super().__init__()
        self.orig = orig
        self.epsilon = epsilon

    def forward(self, x):
        with torch.no_grad():
            w = torch.softmax(self.orig.attention(x), dim=1)  # (B, T, 1)
        return cp_weighted_pool(x, w, self.epsilon)


class LRPAttentionLayer(nn.Module):
    """CP-LRP for ``seqsleepnet.AttentionLayer`` (additive/Bahdanau pooling with
    a manual softmax)."""

    def __init__(self, orig: nn.Module, epsilon: float = 1e-6):
        super().__init__()
        self.orig = orig
        self.epsilon = epsilon

    def forward(self, x, r_alphas: bool = False):
        with torch.no_grad():
            B, S, H = x.size()
            v = torch.tanh(
                torch.matmul(x.reshape(B * S, H), self.orig.W_omega)
                + self.orig.b_omega.reshape(1, -1)
            )
            vu = torch.matmul(v, self.orig.u_omega.reshape(-1, 1))
            exps = torch.exp(vu).reshape(-1, S)
            alphas = (exps / exps.sum(1, keepdim=True)).reshape(B, S, 1)
        out = cp_weighted_pool(x, alphas, self.epsilon)
        if r_alphas:
            return out, alphas.reshape(B, S)
        return out
