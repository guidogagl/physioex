"""Instrumented attention LRP (CP-LRP) for transformer blocks.

Wrapping a stock ``nn.TransformerEncoderLayer`` with a single LRP rule does NOT
conserve relevance (verified empirically), and the layer's forward introspects
``self.self_attn.batch_first`` so its ``MultiheadAttention`` cannot be wrapped
either.  Faithful attention LRP therefore re-expresses the attention/FFN
forward with relevance-carrying primitives:

* linear projections / FFN     → ε-LRP (:func:`~._functional.linear_eps`)
* ``softmax(QKᵀ/√d)·V``          → **CP-LRP** (:class:`_CPAttention`): the
  attention matrix is treated as constant and relevance flows through the
  *value* path (Ali et al. 2022, *XAI for Transformers: Better Explanations
  through Conservative Propagation*).  It conserves exactly; propagating
  through the softmax (full AttnLRP, Achtibat et al. 2024) is non-conserving by
  design and is not implemented here.  Consequence: features that act only
  through *where to attend* (the query/key path) receive zero relevance.
* residual sums                → proportional split (:func:`~._functional.add_eps`)
* LayerNorm                    → **identity rule** (relevance passes through)
* GELU/ReLU                    → **identity rule** (straight-through)

:class:`LRPMultiheadAttention` / :class:`LRPTransformerEncoderLayer` /
:class:`LRPTransformerEncoder` load a trained module's weights via
``from_torch`` (the borrowed leaves are deep-copied and frozen, so the source
model is left untouched); the forward is numerically identical to the fused
module.  Not supported (raise): ``batch_first=False``, ``bias_k``/
``add_zero_attn``, separate q/k/v projection dims, attention masks.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from physioex.explain.lrp._functional import add_eps, linear_eps, stabilize, st_identity


def _ln_identity(layer_norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm with the identity rule: true normalised value, relevance passes."""
    return st_identity(x, layer_norm(x))


class _CPAttention(torch.autograd.Function):
    """Conservative-propagation attention (CP-LRP).

    Forward is the standard ``softmax(QKᵀ·scale) · V``.  In the backward the
    attention matrix is **constant**, so ``ctx = A·V`` is linear in ``V`` and
    ε-LRP gives ``R_v = v ⊙ (Aᵀ · R/(ctx ± ε))``; the query/key path receives
    zero relevance.  Conserves exactly (up to ε).
    """

    @staticmethod
    def forward(ctx, q, k, v, scale, epsilon):
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        ctx.save_for_backward(attn, v, out)
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, relevance):
        attn, v, out = ctx.saved_tensors
        s = relevance / stabilize(out, ctx.epsilon)  # (…, Tq, d)
        r_v = v * torch.matmul(attn.transpose(-2, -1), s)  # (…, Tk, d)
        return None, None, r_v, None, None


def _cp_attention(q, k, v, scale, epsilon=1e-6):
    return _CPAttention.apply(q, k, v, scale, epsilon)


class LRPMultiheadAttention(nn.Module):
    """LRP-instrumented multi-head attention (self or cross), value-path CP-LRP.

    Reproduces ``nn.MultiheadAttention`` (``batch_first=True``, same q/k/v
    embed dims, ``need_weights=False``).  ``forward(query, key, value)`` returns
    the attention output tensor.
    """

    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = float(epsilon)
        self.batch_first = True
        self._p = nn.ParameterDict()

    @classmethod
    def from_torch(cls, mha: nn.MultiheadAttention, epsilon: float = 1e-6):
        if not getattr(mha, "_qkv_same_embed_dim", True):
            raise NotImplementedError("LRPMultiheadAttention: q/k/v embed dims must match")
        if not mha.batch_first:
            raise NotImplementedError("LRPMultiheadAttention: batch_first=False is not supported")
        if mha.bias_k is not None or mha.add_zero_attn:
            raise NotImplementedError("LRPMultiheadAttention: bias_k/add_zero_attn unsupported")
        obj = cls(mha.embed_dim, mha.num_heads, epsilon)

        def frozen(t):
            return nn.Parameter(t.detach().clone(), requires_grad=False)

        obj._p["in_proj_weight"] = frozen(mha.in_proj_weight)
        if mha.in_proj_bias is not None:
            obj._p["in_proj_bias"] = frozen(mha.in_proj_bias)
        obj._p["out_proj_weight"] = frozen(mha.out_proj.weight)
        if mha.out_proj.bias is not None:
            obj._p["out_proj_bias"] = frozen(mha.out_proj.bias)
        return obj

    def _heads(self, t):  # (B, T, E) -> (B, H, T, d)
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def attention_weights(self, query, key):
        """Averaged-over-heads attention matrix ``(B, Tq, Tk)`` (no grad)."""
        with torch.no_grad():
            q, k = self._proj(query, key, key)[:2]
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            return torch.softmax(scores, dim=-1).mean(1)

    def _proj(self, query, key, value):
        E, eps = self.embed_dim, self.epsilon
        W = self._p["in_proj_weight"]
        b = self._p.get("in_proj_bias")
        wq, wk, wv = W[:E], W[E : 2 * E], W[2 * E :]
        bq, bk, bv = (b[:E], b[E : 2 * E], b[2 * E :]) if b is not None else (None,) * 3
        return (
            self._heads(linear_eps(query, wq, bq, eps)),
            self._heads(linear_eps(key, wk, bk, eps)),
            self._heads(linear_eps(value, wv, bv, eps)),
        )

    def forward(self, query, key, value):
        q, k, v = self._proj(query, key, value)
        ctx = _cp_attention(q, k, v, 1.0 / math.sqrt(self.head_dim), self.epsilon)
        B, _, Tq, _ = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, Tq, self.embed_dim)
        return linear_eps(
            ctx, self._p["out_proj_weight"], self._p.get("out_proj_bias"), self.epsilon
        )


class LRPMultiheadAttentionModule(nn.Module):
    """``nn.MultiheadAttention``-compatible adapter for *standalone* attention.

    Host models call ``out, w = mha(query=, key=, value=, ...)``.  Returns
    ``(out, weights_or_None)``; raises on attention masks (unsupported) instead
    of silently ignoring them.
    """

    def __init__(self, attn: LRPMultiheadAttention):
        super().__init__()
        self.attn = attn
        self.batch_first = True

    @classmethod
    def from_torch(cls, mha: nn.MultiheadAttention, epsilon: float = 1e-6):
        return cls(LRPMultiheadAttention.from_torch(mha, epsilon))

    def forward(self, query, key, value, need_weights=True, **kwargs):
        for name in ("attn_mask", "key_padding_mask"):
            if kwargs.get(name) is not None:
                raise NotImplementedError(f"LRP attention does not support {name}")
        if kwargs.get("is_causal"):
            raise NotImplementedError("LRP attention does not support is_causal=True")
        out = self.attn(query, key, value)
        weights = self.attn.attention_weights(query, key) if need_weights else None
        return out, weights


class LRPTransformerEncoderLayer(nn.Module):
    """LRP-instrumented ``nn.TransformerEncoderLayer`` (post- and pre-norm).

    Build with :meth:`from_torch`.  Self-attention only; ``src_mask`` /
    ``src_key_padding_mask`` are not supported (raise).
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.self_attn = None
        self.linear1 = self.linear2 = self.norm1 = self.norm2 = None
        self.activation = torch.nn.functional.relu
        self.norm_first = False
        self.epsilon = float(epsilon)

    @classmethod
    def from_torch(cls, tel: nn.TransformerEncoderLayer, epsilon: float = 1e-6):
        obj = cls(epsilon)
        obj.self_attn = LRPMultiheadAttention.from_torch(tel.self_attn, epsilon)
        for name in ("linear1", "linear2", "norm1", "norm2"):
            leaf = copy.deepcopy(getattr(tel, name)).eval()
            for p in leaf.parameters():
                p.requires_grad_(False)
            setattr(obj, name, leaf)
        obj.activation = tel.activation
        obj.norm_first = bool(getattr(tel, "norm_first", False))
        return obj

    def _ff(self, x):
        eps = self.epsilon
        h = linear_eps(x, self.linear1.weight, self.linear1.bias, eps)
        h = st_identity(h, self.activation(h))  # identity rule for the nonlinearity
        return linear_eps(h, self.linear2.weight, self.linear2.bias, eps)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_mask is not None or src_key_padding_mask is not None or is_causal:
            raise NotImplementedError("LRPTransformerEncoderLayer does not support masks")
        eps = self.epsilon
        if self.norm_first:
            x = add_eps(x, self.self_attn(*(_ln_identity(self.norm1, x),) * 3), eps)
            x = add_eps(x, self._ff(_ln_identity(self.norm2, x)), eps)
        else:
            x = _ln_identity(self.norm1, add_eps(x, self.self_attn(x, x, x), eps))
            x = _ln_identity(self.norm2, add_eps(x, self._ff(x), eps))
        return x


class LRPTransformerEncoder(nn.Module):
    """LRP replacement for ``nn.TransformerEncoder`` (the container's own forward
    introspects ``layers[0].self_attn.batch_first``, so the whole stack is
    replaced).  Applies the LRP layers in turn, then the optional final norm."""

    def __init__(self, layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    @classmethod
    def from_torch(cls, encoder: nn.TransformerEncoder, epsilon: float = 1e-6):
        layers = [LRPTransformerEncoderLayer.from_torch(l, epsilon) for l in encoder.layers]
        norm = getattr(encoder, "norm", None)
        if norm is not None:
            norm = copy.deepcopy(norm).eval()
            for p in norm.parameters():
                p.requires_grad_(False)
        return cls(layers, norm)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        if mask is not None or src_key_padding_mask is not None:
            raise NotImplementedError("LRPTransformerEncoder does not support masks")
        for layer in self.layers:
            x = layer(x)
        return _ln_identity(self.norm, x) if self.norm is not None else x


def swap_transformer_layers(module: nn.Module, epsilon: float = 1e-6) -> nn.Module:
    """In-place: replace ``nn.TransformerEncoder`` / ``nn.TransformerEncoderLayer``
    children with their LRP counterparts (weights copied; the source leaves are
    not modified).  Returns ``module``.  Prefer :func:`~.model.prepare_model_for_lrp`."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.TransformerEncoder):
            setattr(module, name, LRPTransformerEncoder.from_torch(child, epsilon))
        elif isinstance(child, nn.TransformerEncoderLayer):
            setattr(module, name, LRPTransformerEncoderLayer.from_torch(child, epsilon))
        elif not isinstance(child, (LRPTransformerEncoder, LRPTransformerEncoderLayer)):
            swap_transformer_layers(child, epsilon)
    return module
