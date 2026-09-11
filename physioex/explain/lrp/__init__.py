"""Layer-wise Relevance Propagation (LRP) for PhysioEx.

Conservation-based attribution (Bach et al. 2015).  Two entry points:

* :class:`LRP` — Zennit composites for feed-forward / CNN models
  (ε, γ, w², z-box, flat; BatchNorm canonization).
* :class:`ModelLRP` — whole-model LRP for architectures with LSTM/GRU
  (Arras signal-take), attention / transformers (CP-LRP, Ali et al. 2022) and
  PhysioEx's softmax poolings; leaves get ε-LRP, norms/activations the identity
  rule, residual ``+`` the proportional rule.

Both seed the target logit so ``Σ R ≈ f_c(x)``; use ``return_report=True`` /
:func:`check_conservation` to see how much biases absorb, and
:func:`audit_lrp_coverage` to list blocks left on plain autograd.

Requires the optional ``explain`` extra::

    pip install "physioex[explain]"
"""

from physioex.explain.lrp._rules import EpsilonRule, IdentityRule
from physioex.explain.lrp._functional import (
    add_eps,
    linear_eps,
    mul_signal_take,
    st_identity,
    target_seed,
)
from physioex.explain.lrp.attributor import LRP
from physioex.explain.lrp.canonizers import default_canonizers
from physioex.explain.lrp.composites import epsilon_composite, physioex_composite
from physioex.explain.lrp.diagnostics import ConservationReport, check_conservation
from physioex.explain.lrp.model import (
    ModelLRP,
    audit_lrp_coverage,
    prepare_model_for_lrp,
    register_lrp_adapter,
)
from physioex.explain.lrp.pooling import (
    LRPAttentionLayer,
    LRPAttentionPooling,
    LRPChannelMixer,
    cp_weighted_pool,
)
from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM
from physioex.explain.lrp.transformer import (
    LRPMultiheadAttention,
    LRPMultiheadAttentionModule,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
    swap_transformer_layers,
)

__all__ = [
    # entry points
    "LRP",
    "ModelLRP",
    # Zennit path
    "physioex_composite",
    "epsilon_composite",
    "default_canonizers",
    # whole-model path
    "prepare_model_for_lrp",
    "register_lrp_adapter",
    "audit_lrp_coverage",
    # diagnostics
    "ConservationReport",
    "check_conservation",
    # blocks
    "LRPLSTM",
    "LRPGRU",
    "LRPMultiheadAttention",
    "LRPMultiheadAttentionModule",
    "LRPTransformerEncoderLayer",
    "LRPTransformerEncoder",
    "swap_transformer_layers",
    "LRPAttentionPooling",
    "LRPAttentionLayer",
    "LRPChannelMixer",
    "cp_weighted_pool",
    # rules & primitives
    "EpsilonRule",
    "IdentityRule",
    "linear_eps",
    "add_eps",
    "mul_signal_take",
    "st_identity",
    "target_seed",
]
