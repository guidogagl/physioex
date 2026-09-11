"""End-to-end LRP for whole PhysioEx models.

:func:`prepare_model_for_lrp` rewrites a model *in place* so that its backward
pass carries relevance:

* ``nn.LSTM`` / ``nn.GRU``                → :class:`LRPLSTM` / :class:`LRPGRU`
* ``nn.TransformerEncoder(Layer)``        → the ``transformer`` LRP versions
* standalone ``nn.MultiheadAttention``    → :class:`LRPMultiheadAttentionModule`
* registered custom blocks (PhysioEx's softmax poolings, the learnable
  filterbank; see :func:`register_lrp_adapter`) → their CP-LRP / ε adapters
* leaf ``Linear`` / ``Conv``              → LXT ``EpsilonRule`` (ε-LRP)
* ``BatchNorm``                           → merged into the preceding linear layer
  (Zennit canonizer), then identity
* ``LayerNorm`` / ``GroupNorm`` / element-wise activations → identity rule

Everything the walk cannot account for is reported by
:func:`audit_lrp_coverage` (parametric leaves left on plain autograd would
propagate *gradient*, not relevance).

:class:`ModelLRP` deep-copies the trained model, prepares it and runs a
target-seeded backward.  Plain ``+`` residuals written in a model's own
``forward`` are redirected to the proportional ``add_eps`` rule at runtime via a
``TorchFunctionMode`` (``patch_residuals=True``); without it a residual gives
both branches the full relevance and total relevance is over-counted.

Requires the ``explain`` extra (``zennit``, ``lxt>=2.0``).
"""

from __future__ import annotations

import contextlib
import copy
import warnings
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from physioex.explain.lrp._functional import add_eps, target_seed
from physioex.explain.lrp.diagnostics import ConservationReport
from physioex.explain.lrp.pooling import (
    LRPAttentionLayer,
    LRPAttentionPooling,
    LRPChannelMixer,
    _PoolAdapter,
)
from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM
from physioex.explain.lrp.transformer import (
    LRPMultiheadAttention,
    LRPMultiheadAttentionModule,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
)

AdapterFactory = Callable[[nn.Module, float], nn.Module]

# ---------------------------------------------------------------------------
# adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS_BY_CLASS: Dict[type, AdapterFactory] = {}


def _epsilon_rule_factory(module: nn.Module, epsilon: float) -> nn.Module:
    """ε-LRP via LXT's vjp super-function — exact for modules linear in their
    input (e.g. ``LearnableFilterbank``: ``x @ (sigmoid(W)·S)``)."""
    from lxt.explicit.rules import EpsilonRule

    return EpsilonRule(module, epsilon)


# PhysioEx blocks, matched by fully-qualified class name (no model import needed
# and no collision with unrelated classes sharing a short name, e.g.
# ``sleepfm.AttentionPooling``).  Adapters with a ``matches`` classmethod are
# additionally checked structurally.
_BUILTIN_BY_QUALNAME: Dict[str, AdapterFactory] = {
    "physioex.models.sleeptransformer.AttentionPooling": LRPAttentionPooling,
    "physioex.models.seqsleepnet.AttentionLayer": LRPAttentionLayer,
    "physioex.models.seqsleepnet.LearnableFilterbank": _epsilon_rule_factory,
    "physioex.models.protosleepnet.ChannelMixer": LRPChannelMixer,
}


def register_lrp_adapter(cls: type, factory: AdapterFactory) -> None:
    """Register ``factory(module, epsilon) -> nn.Module`` for instances of ``cls``.

    Use for custom blocks whose forward contains softmax pooling / attention
    or other non-module ops that plain autograd would attribute as gradient.
    """
    _ADAPTERS_BY_CLASS[cls] = factory


def _adapter_for(module: nn.Module) -> Optional[AdapterFactory]:
    for cls, factory in _ADAPTERS_BY_CLASS.items():
        if isinstance(module, cls):
            return factory
    qualname = f"{type(module).__module__}.{type(module).__qualname__}"
    factory = _BUILTIN_BY_QUALNAME.get(qualname)
    if factory is not None:
        matches = getattr(factory, "matches", None)
        if matches is not None and not matches(module):
            return None
    return factory


# ---------------------------------------------------------------------------
# module classification
# ---------------------------------------------------------------------------

_LRP_TYPES = (
    LRPLSTM,
    LRPGRU,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
    LRPMultiheadAttention,
    LRPMultiheadAttentionModule,
    _PoolAdapter,
)
_ACTIVATIONS = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid,
    nn.Softplus, nn.Hardswish, nn.Hardsigmoid, nn.Mish, nn.PReLU,
)
_NORMS_IDENTITY = (nn.LayerNorm, nn.GroupNorm) + (
    (nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ()
)
_LINEAR_LEAVES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BATCHNORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
# parametric leaves that are constant w.r.t. the input and need no rule
_CONSTANT_LEAVES = (nn.Embedding,)


def _is_lrp_module(m: nn.Module) -> bool:
    try:
        from lxt.explicit.rules import WrapModule
    except ImportError:  # pragma: no cover
        WrapModule = ()
    return isinstance(m, _LRP_TYPES) or isinstance(m, WrapModule)


def _bn_is_identity(bn: nn.Module) -> bool:
    w_ok = bn.weight is None or torch.allclose(bn.weight, torch.ones_like(bn.weight))
    b_ok = bn.bias is None or torch.allclose(bn.bias, torch.zeros_like(bn.bias))
    m_ok = torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean))
    v_ok = torch.allclose(bn.running_var, torch.ones_like(bn.running_var))
    return bool(w_ok and b_ok and m_ok and v_ok)


def _merge_batchnorm(model: nn.Module):
    """Fold sequential BatchNorm layers into the preceding linear/conv (Zennit
    canonizer).  Returns the handles (keep them alive to keep the merge)."""
    try:
        from zennit.canonizers import SequentialMergeBatchNorm
    except ImportError:
        return []
    return SequentialMergeBatchNorm().apply(model)


# ---------------------------------------------------------------------------
# preparation walk
# ---------------------------------------------------------------------------


def _replace(child: nn.Module, epsilon: float):
    """Return the LRP replacement for ``child`` or ``None`` to recurse into it."""
    from lxt.explicit.rules import EpsilonRule, IdentityRule

    if _is_lrp_module(child):
        return child  # already prepared (idempotence)
    if isinstance(child, nn.LSTM):
        return LRPLSTM.from_torch(child, epsilon)
    if isinstance(child, nn.GRU):
        return LRPGRU.from_torch(child, epsilon)
    if isinstance(child, nn.TransformerEncoder):
        return LRPTransformerEncoder.from_torch(child, epsilon)
    if isinstance(child, nn.TransformerEncoderLayer):
        return LRPTransformerEncoderLayer.from_torch(child, epsilon)
    if isinstance(child, nn.MultiheadAttention):
        return LRPMultiheadAttentionModule.from_torch(child, epsilon)
    factory = _adapter_for(child)
    if factory is not None:
        return factory(child, epsilon)
    if isinstance(child, _LINEAR_LEAVES):
        return EpsilonRule(child, epsilon)
    if isinstance(child, _BATCHNORMS):
        # after canonization BN is the identity; otherwise treat it as the
        # affine-linear map it is in eval mode (its shift leaks like a bias)
        return IdentityRule(child) if _bn_is_identity(child) else EpsilonRule(child, epsilon)
    if isinstance(child, _NORMS_IDENTITY) or isinstance(child, _ACTIVATIONS):
        return IdentityRule(child)
    return None


def prepare_model_for_lrp(
    model: nn.Module,
    epsilon: float = 1e-6,
    strict: bool = False,
    _memo: Optional[Dict[int, nn.Module]] = None,
) -> nn.Module:
    """In-place: swap fused / attention / registered blocks for their LRP
    counterparts and wrap the remaining leaves with LRP rules.  Returns
    ``model``.  Call on a copy (see :class:`ModelLRP`).

    ``strict=True`` raises if any parametric leaf is left on plain autograd.
    Shared modules (the same object under several attributes) are replaced by
    a single shared LRP module.
    """
    top = _memo is None
    memo: Dict[int, nn.Module] = {} if top else _memo
    if top:
        model._lrp_bn_handles = _merge_batchnorm(model)
    for name, child in list(model._modules.items()):
        if child is None:
            continue
        key = id(child)
        if key in memo:
            setattr(model, name, memo[key])
            continue
        repl = _replace(child, epsilon)
        if repl is None:
            prepare_model_for_lrp(child, epsilon, strict, memo)
            memo[key] = child
            continue
        memo[key] = repl
        setattr(model, name, repl)
        if isinstance(repl, _PoolAdapter):
            # the adapter's inner blocks (e.g. ChannelMixer.mixer) still need rules
            prepare_model_for_lrp(repl.orig, epsilon, strict, memo)
    if top:
        uncovered = audit_lrp_coverage(model)
        if uncovered:
            msg = (
                "LRP coverage: these parametric leaves are not wrapped by an LRP "
                "rule and will propagate gradient instead of relevance: "
                + ", ".join(uncovered)
            )
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return model


def audit_lrp_coverage(model: nn.Module, include_containers: bool = False) -> List[str]:
    """Names of parametric **leaf** modules that carry no LRP rule (their forward
    runs on plain autograd and propagates gradient, not relevance).

    Container modules that own parameters *and* have children (e.g. a learned
    CLS token or positional embedding added to the stream) are additive
    constants w.r.t. the input and are only listed with
    ``include_containers=True`` for manual inspection.
    """
    uncovered: List[str] = []

    def walk(m: nn.Module, prefix: str, skip_own: bool = False):
        if _is_lrp_module(m):
            if isinstance(m, _PoolAdapter):
                # the adapter uses orig's own parameters as constants (no_grad);
                # only orig's inner blocks still need rules
                walk(m.orig, prefix + ".orig", skip_own=True)
            return
        own = (not skip_own) and any(True for _ in m.parameters(recurse=False))
        children = list(m.named_children())
        if own and not isinstance(m, _CONSTANT_LEAVES):
            if not children or include_containers:
                uncovered.append(prefix or type(m).__name__)
        for n, c in children:
            walk(c, f"{prefix}.{n}" if prefix else n)

    walk(model, "")
    return uncovered


# ---------------------------------------------------------------------------
# residual add → proportional rule, at runtime
# ---------------------------------------------------------------------------


class _ResidualAddMode(torch.overrides.TorchFunctionMode):
    """Redirect ``a + b`` (both requiring grad, grad enabled) to ``add_eps``.

    Custom autograd Functions run their forward with grad disabled, so the
    adds *inside* the LRP rules are never intercepted (no recursion)."""

    _ADD_FUNCS = (torch.add, torch.Tensor.add, torch.Tensor.__add__, torch.Tensor.__radd__)

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func in self._ADD_FUNCS
            and not kwargs
            and len(args) == 2
            and torch.is_grad_enabled()
            and all(isinstance(a, torch.Tensor) and a.requires_grad for a in args)
        ):
            return add_eps(args[0], args[1], self.epsilon)
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# attributor
# ---------------------------------------------------------------------------


class ModelLRP(nn.Module):
    """LRP attribution for a whole PhysioEx model.

    Args:
        model: trained model ``(B, L, C, T[, F]) -> (B, L, n_classes)`` (or a
            dict — set ``output_key``).  Deep-copied; the original is untouched.
        in_index / out_index: sequence epoch and class logit to explain.
        output_key: for dict outputs (CoReSleep → ``"combined"``).
        epsilon: ε of every ε-rule in the prepared model.
        patch_residuals: redirect plain ``+`` between grad-carrying tensors in
            the model's own forward to the proportional rule (recommended).
        strict: raise if a parametric leaf is left without an LRP rule.
        copy_model: set ``False`` to prepare ``model`` in place (saves memory).

    ``forward(x)`` returns relevance shaped like ``x`` (seeded with the target
    logit so ``Σ R ≈ f_c(x)``, minus what biases absorb);
    ``forward(x, return_report=True)`` also returns a
    :class:`~physioex.explain.lrp.diagnostics.ConservationReport`.
    """

    def __init__(
        self,
        model: nn.Module,
        in_index: int = 0,
        out_index: int = 0,
        output_key: Optional[str] = None,
        epsilon: float = 1e-6,
        patch_residuals: bool = True,
        strict: bool = False,
        copy_model: bool = True,
    ):
        super().__init__()
        self.in_index = in_index
        self.out_index = out_index
        self.output_key = output_key
        self.epsilon = float(epsilon)
        self.patch_residuals = patch_residuals
        prepared = copy.deepcopy(model) if copy_model else model
        prepared.eval()
        for m in prepared.modules():
            if getattr(m, "inplace", False):
                m.inplace = False  # in-place ops would corrupt saved tensors
        for p in prepared.parameters():
            p.requires_grad_(False)
        self.model = prepare_model_for_lrp(prepared, epsilon=self.epsilon, strict=strict)
        for p in self.model.parameters():  # BN merging may create new bias params
            p.requires_grad_(False)
        self.uncovered = audit_lrp_coverage(self.model)

    def _output(self, out):
        return out[self.output_key] if self.output_key is not None else out

    def forward(self, x: torch.Tensor, return_report: bool = False):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            mode = _ResidualAddMode(self.epsilon) if self.patch_residuals else contextlib.nullcontext()
            with mode:
                out = self._output(self.model(x))
            seed, target = target_seed(out, self.in_index, self.out_index)
            (relevance,) = torch.autograd.grad(out, x, seed)
        if return_report:
            return relevance, ConservationReport.from_relevance(target, relevance)
        return relevance
