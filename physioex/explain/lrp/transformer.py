"""Instrumented AttnLRP for attention / transformer blocks.

Wrapping a stock ``nn.TransformerEncoderLayer`` with a single LRP rule does NOT
conserve relevance (verified empirically), and the layer's forward introspects
``self.self_attn.batch_first`` so its ``MultiheadAttention`` cannot be wrapped
either.  Faithful attention LRP (AttnLRP, Achtibat et al., ICML 2024) therefore
requires re-expressing the attention/FFN forward with the LXT functionals so the
backward carries relevance:

* linear projections / FFN     → ``lxt.explicit.functional.linear_epsilon``
* ``QKᵀ`` and ``A·V`` products  → ``lxt.explicit.functional.matmul``
* attention softmax            → ``lxt.explicit.functional.softmax``
* residual sums                → ``lxt.explicit.functional.add2`` (proportional)
* LayerNorm                    → **identity rule** (relevance passes through)
* GELU/ReLU                    → standard activation (autograd handles the mask)

:class:`LRPMultiheadAttention` and :class:`LRPTransformerEncoderLayer` load a
trained ``nn.MultiheadAttention`` / ``nn.TransformerEncoderLayer`` unchanged
(``from_torch``); the forward is numerically identical to the fused module.

Requires the ``explain`` extra (``lxt``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ln_identity(layer_norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm with the AttnLRP identity rule: forward is the true normalised
    value, backward passes relevance through unchanged."""
    y = layer_norm(x)
    return x + (y - x).detach()


class _CPAttention(torch.autograd.Function):
    """Conservative-propagation attention (CP-LRP).

    Forward is the standard ``softmax(QKᵀ·scale) · V``.  In the backward, the
    attention matrix is treated as **constant** (a learned gating), so relevance
    flows entirely through the **value** path via ε-LRP (``ctx = A·V`` is then
    linear in V), and the query/key path receives zero.  This conserves relevance
    exactly (up to ε) — unlike propagating through the softmax, which does not.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale, epsilon):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        ctx.save_for_backward(attn, v, out)
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, relevance):
        attn, v, out = ctx.saved_tensors
        eps = ctx.epsilon
        denom = out + torch.where(out >= 0, eps, -eps)
        s = relevance / denom              # (…, Tq, d)
        # ε-LRP through the linear map ctx = A·V (A constant):
        #   R(v[t,d]) = v[t,d] · Σ_i A[i,t] · s[i,d] = v ⊙ (Aᵀ s)
        r_v = v * torch.matmul(attn.transpose(-2, -1), s)
        # q, k receive zero relevance (CP-LRP); scale/epsilon are non-tensor.
        return None, None, r_v, None, None


def _cp_attention(q, k, v, scale, epsilon=1e-6):
    """``softmax(QKᵀ·scale)·V`` with the CP-LRP backward (relevance → V only)."""
    return _CPAttention.apply(q, k, v, scale, epsilon)


class LRPMultiheadAttention(nn.Module):
    """LRP-instrumented multi-head attention (self or cross).

    Reproduces ``nn.MultiheadAttention`` (``_qkv_same_embed_dim`` case,
    ``need_weights=False``) using LXT functionals.  Build with
    :meth:`from_torch`.  ``forward(query, key, value)`` returns the attention
    output tensor (same shape as ``query``); for self-attention pass the same
    tensor three times.
    """

    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        # harmless attributes some containers introspect for fast-paths
        self.batch_first = True
        self._p = nn.ParameterDict()

    @classmethod
    def from_torch(cls, mha: nn.MultiheadAttention) -> "LRPMultiheadAttention":
        if not getattr(mha, "_qkv_same_embed_dim", True):
            raise NotImplementedError(
                "LRPMultiheadAttention supports the qkv-same-embed-dim case only."
            )
        obj = cls(mha.embed_dim, mha.num_heads)
        obj._p["in_proj_weight"] = nn.Parameter(
            mha.in_proj_weight.detach().clone(), requires_grad=False
        )
        if mha.in_proj_bias is not None:
            obj._p["in_proj_bias"] = nn.Parameter(
                mha.in_proj_bias.detach().clone(), requires_grad=False
            )
        obj._p["out_proj_weight"] = nn.Parameter(
            mha.out_proj.weight.detach().clone(), requires_grad=False
        )
        if mha.out_proj.bias is not None:
            obj._p["out_proj_bias"] = nn.Parameter(
                mha.out_proj.bias.detach().clone(), requires_grad=False
            )
        return obj

    def _heads(self, t):  # (B, T, E) -> (B, H, T, d)
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value):
        from lxt.explicit.functional import linear_epsilon

        E = self.embed_dim
        W = self._p["in_proj_weight"]
        b = self._p.get("in_proj_bias")
        wq, wk, wv = W[:E], W[E : 2 * E], W[2 * E :]
        bq, bk, bv = (b[:E], b[E : 2 * E], b[2 * E :]) if b is not None else (None,) * 3

        q = linear_epsilon(query, wq, bq, epsilon=self.epsilon)
        k = linear_epsilon(key, wk, bk, epsilon=self.epsilon)
        v = linear_epsilon(value, wv, bv, epsilon=self.epsilon)

        q = self._heads(q)  # (B, H, Tq, d)
        k = self._heads(k)
        v = self._heads(v)

        # CP-LRP attention: softmax(QKᵀ/√d)·V with relevance routed through V.
        ctx = _cp_attention(q, k, v, 1.0 / math.sqrt(self.head_dim), self.epsilon)

        B, _, Tq, _ = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, Tq, E)
        return linear_epsilon(
            ctx, self._p["out_proj_weight"], self._p.get("out_proj_bias"),
            epsilon=self.epsilon,
        )


class LRPMultiheadAttentionModule(nn.Module):
    """``nn.MultiheadAttention``-compatible wrapper for standalone attention.

    Host models call ``nn.MultiheadAttention`` as ``out, _ = mha(query=, key=,
    value=, ...)`` — a 2-tuple return with keyword args.  This adapter mirrors
    that interface around :class:`LRPMultiheadAttention` (which itself returns a
    bare tensor for the transformer-layer internals), returning ``(out, None)``
    and ignoring the extra kwargs (need_weights, attn_mask, …).
    """

    def __init__(self, attn: "LRPMultiheadAttention"):
        super().__init__()
        self.attn = attn
        self.batch_first = True

    @classmethod
    def from_torch(cls, mha: nn.MultiheadAttention) -> "LRPMultiheadAttentionModule":
        return cls(LRPMultiheadAttention.from_torch(mha))

    def forward(self, query, key, value, **kwargs):
        return self.attn(query, key, value), None


class LRPTransformerEncoderLayer(nn.Module):
    """LRP-instrumented ``nn.TransformerEncoderLayer`` (post- and pre-norm).

    Build with :meth:`from_torch`; forward matches the fused layer while the
    backward carries relevance.  Self-attention only (no ``src_mask`` /
    ``src_key_padding_mask`` — these models don't use them).
    """

    def __init__(self, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = None
        self.linear1 = None
        self.linear2 = None
        self.norm1 = None
        self.norm2 = None
        self.activation = F.relu
        self.norm_first = False
        self.epsilon = 1e-6

    @classmethod
    def from_torch(cls, tel: nn.TransformerEncoderLayer) -> "LRPTransformerEncoderLayer":
        obj = cls()
        obj.self_attn = LRPMultiheadAttention.from_torch(tel.self_attn)
        obj.linear1 = tel.linear1
        obj.linear2 = tel.linear2
        obj.norm1 = tel.norm1
        obj.norm2 = tel.norm2
        obj.activation = tel.activation
        obj.norm_first = getattr(tel, "norm_first", False)
        # freeze the borrowed leaf modules (attribution over a trained model)
        for m in (obj.linear1, obj.linear2, obj.norm1, obj.norm2):
            for p in m.parameters():
                p.requires_grad_(False)
        return obj

    def _sa(self, x):
        return self.self_attn(x, x, x)

    def _ff(self, x):
        from lxt.explicit.functional import linear_epsilon

        h = linear_epsilon(x, self.linear1.weight, self.linear1.bias, epsilon=self.epsilon)
        h = self.activation(h)
        return linear_epsilon(h, self.linear2.weight, self.linear2.bias, epsilon=self.epsilon)

    def forward(self, x):
        from lxt.explicit.functional import add2

        if self.norm_first:
            x = add2(x, self._sa(_ln_identity(self.norm1, x)))
            x = add2(x, self._ff(_ln_identity(self.norm2, x)))
        else:
            x = _ln_identity(self.norm1, add2(x, self._sa(x)))
            x = _ln_identity(self.norm2, add2(x, self._ff(x)))
        return x


class LRPTransformerEncoder(nn.Module):
    """LRP replacement for ``nn.TransformerEncoder``.

    ``nn.TransformerEncoder``'s own forward introspects
    ``layers[0].self_attn.batch_first`` for a nested-tensor fast path, so the
    whole stack is replaced rather than only its layers.  Simply applies each
    :class:`LRPTransformerEncoderLayer` in turn, then the optional final norm
    (identity rule).
    """

    def __init__(self, layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    @classmethod
    def from_torch(cls, encoder: nn.TransformerEncoder) -> "LRPTransformerEncoder":
        layers = [LRPTransformerEncoderLayer.from_torch(l) for l in encoder.layers]
        norm = getattr(encoder, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(False)
        return cls(layers, norm)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = _ln_identity(self.norm, x)
        return x


def swap_transformer_layers(module: nn.Module) -> nn.Module:
    """In-place: replace every ``nn.TransformerEncoder`` /
    ``nn.TransformerEncoderLayer`` under ``module`` with its LRP counterpart
    (:class:`LRPTransformerEncoder` / :class:`LRPTransformerEncoderLayer`)
    sharing the trained weights.  Returns ``module``."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.TransformerEncoder):
            setattr(module, name, LRPTransformerEncoder.from_torch(child))
        elif isinstance(child, nn.TransformerEncoderLayer):
            setattr(module, name, LRPTransformerEncoderLayer.from_torch(child))
        else:
            swap_transformer_layers(child)
    return module
