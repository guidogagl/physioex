"""Module-level LRP rule wrappers (relevance-as-gradient convention).

Small, dependency-free counterparts of LXT's ``EpsilonRule`` / ``IdentityRule``
module wrappers: they replace a module in the tree, reproduce its forward
exactly, and substitute the backward with the LRP rule.

* :class:`EpsilonRule` — ε-LRP through a module that is **linear in its input**
  (``Linear``, ``Conv*``, eval-mode ``BatchNorm``, PhysioEx's
  ``LearnableFilterbank``): ``R_x = x ⊙ Jᵀ(R_y / (y ± ε))`` computed with a
  vector-Jacobian product, so any such module is handled without knowing its
  weights layout.  Bias relevance is absorbed (standard LRP-ε).
* :class:`IdentityRule` — relevance passes through unchanged (LayerNorm,
  GroupNorm, element-wise activations; shape-preserving modules only).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physioex.explain.lrp._functional import stabilize, st_identity


class _RuleWrapper(nn.Module):
    """Base class: holds the wrapped ``module`` (frozen)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in module.parameters():
            p.requires_grad_(False)

    def extra_repr(self) -> str:  # pragma: no cover - repr only
        return ""


class _EpsilonVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, epsilon):
        y = module(x)
        ctx.module, ctx.epsilon = module, epsilon
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, relevance):
        x, y = ctx.saved_tensors
        s = relevance / stabilize(y, ctx.epsilon)
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(ctx.module(x_), x_, s)
        return x * grad, None, None


class EpsilonRule(_RuleWrapper):
    """ε-LRP wrapper for a single-input module linear in its input."""

    def __init__(self, module: nn.Module, epsilon: float = 1e-6):
        super().__init__(module)
        self.epsilon = float(epsilon)

    def forward(self, x):
        return _EpsilonVJP.apply(x, self.module, self.epsilon)


class IdentityRule(_RuleWrapper):
    """Identity-rule wrapper: forward is the module's, backward passes relevance."""

    def forward(self, x, *args, **kwargs):
        return st_identity(x, self.module(x, *args, **kwargs))
