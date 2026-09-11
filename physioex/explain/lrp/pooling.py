"""CP-LRP for softmax attention-pooling blocks.

PhysioEx's custom poolings collapse a sequence with softmax weights,
``out = Σ_t softmax(score(x))_t · x_t``: ``AttentionPooling`` (sleeptransformer),
``AttentionLayer`` (seqsleepnet; reused by lseqsleepnet and protosleepnet) and
``ChannelMixer`` (protosleepnet, over the channel axis).  Their softmax leaks
relevance just like attention, so the conservative-propagation rule applies:
the pooling weights are treated as **constant** (computed under ``no_grad``
from the original module's parameters) and relevance flows through the value
path, which conserves exactly.  The adapters keep the original module as
``orig`` and reproduce its forward bit-for-bit.

Register additional pooling classes with
:func:`~physioex.explain.lrp.model.register_lrp_adapter`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.explain.lrp._functional import add_eps, stabilize


class _CPWeightedPool(torch.autograd.Function):
    """``out = Σ_t w_t · x_t`` over ``dim=1``, weights constant:
    ``R(x[t,d]) = x[t,d] · w[t] · R_out[d] / (out[d] ± ε)``."""

    @staticmethod
    def forward(ctx, x, weights, epsilon):
        out = (x * weights).sum(dim=1)
        ctx.save_for_backward(x, weights, out)
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, relevance):
        x, weights, out = ctx.saved_tensors
        s = (relevance / stabilize(out, ctx.epsilon)).unsqueeze(1)  # (B, 1, D)
        return x * weights * s, None, None


def cp_weighted_pool(x, weights, epsilon=1e-6):
    """Weighted sum over ``dim=1`` with the CP-LRP backward; ``weights`` is
    ``(B, T, 1)`` and must be detached (constant)."""
    return _CPWeightedPool.apply(x, weights, epsilon)


class _PoolAdapter(nn.Module):
    def __init__(self, orig: nn.Module, epsilon: float = 1e-6):
        super().__init__()
        self.orig = orig
        self.epsilon = float(epsilon)


class LRPAttentionPooling(_PoolAdapter):
    """CP-LRP for ``sleeptransformer.AttentionPooling`` (``self.attention`` MLP,
    softmax over ``dim=1``, ``(B,T,D) -> (B,D)``)."""

    @staticmethod
    def matches(module: nn.Module) -> bool:
        return isinstance(getattr(module, "attention", None), nn.Module)

    def forward(self, x):
        with torch.no_grad():
            w = torch.softmax(self.orig.attention(x), dim=1)  # (B, T, 1)
        return cp_weighted_pool(x, w, self.epsilon)


class LRPAttentionLayer(_PoolAdapter):
    """CP-LRP for ``seqsleepnet.AttentionLayer`` (additive pooling, manual softmax)."""

    @staticmethod
    def matches(module: nn.Module) -> bool:
        return all(hasattr(module, n) for n in ("W_omega", "b_omega", "u_omega"))

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


class LRPChannelMixer(_PoolAdapter):
    """CP-LRP for ``protosleepnet.ChannelMixer``: constant modality embedding →
    per-channel ``mcy`` logits → dropout (eval) → residual ``x + mixer(x)``
    (proportional split; the mixer itself is swapped by ``prepare``) → softmax
    channel pooling (CP-LRP).  ``forward(x, zero_emb) -> (h, mcy_logits)``."""

    @staticmethod
    def matches(module: nn.Module) -> bool:
        return all(
            hasattr(module, n) for n in ("modality_emb", "mcy", "mixer", "attn_pool", "dropout")
        )

    def forward(self, x, zero_emb):
        o = self.orig
        BL, C, d = x.shape
        x = x + o.modality_emb(torch.arange(C, device=x.device)).unsqueeze(0)
        mcy_logits = o.mcy(x)
        x = o.dropout(x, zero_emb, o.channels_acc)
        x = add_eps(x, o.mixer(x), self.epsilon)
        with torch.no_grad():
            w = F.softmax(o.attn_pool(x), dim=1)  # (BL, C, 1)
        return cp_weighted_pool(x, w, self.epsilon), mcy_logits
