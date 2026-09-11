"""Layer-wise Relevance Propagation (LRP) for PhysioEx.

Conservation-based attribution (Bach et al. 2015) built on **Zennit** (CNN/RNN
rules, canonizers) and — for attention + LayerNorm in the transformer and
foundation encoders — **LXT / AttnLRP** (Achtibat et al., ICML 2024).

Requires the optional ``explain`` extra::

    pip install "physioex[explain]"

Public API:
    - :class:`LRP` — the attributor (wraps a model, returns relevance).
    - :func:`physioex_composite` / :func:`epsilon_composite` — rule composites.
    - :func:`default_canonizers` — BatchNorm canonizers.
"""

from physioex.explain.lrp.attributor import LRP
from physioex.explain.lrp.canonizers import default_canonizers
from physioex.explain.lrp.composites import epsilon_composite, physioex_composite
from physioex.explain.lrp.model import ModelLRP, prepare_model_for_lrp
from physioex.explain.lrp.pooling import LRPAttentionLayer, LRPAttentionPooling
from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM
from physioex.explain.lrp.transformer import (
    LRPMultiheadAttention,
    LRPMultiheadAttentionModule,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
    swap_transformer_layers,
)

__all__ = [
    # CNN / Zennit composites (Phase 1)
    "LRP",
    "physioex_composite",
    "epsilon_composite",
    "default_canonizers",
    # recurrent (Phase 2a)
    "LRPLSTM",
    "LRPGRU",
    # transformer / attention (Phase 2b)
    "LRPMultiheadAttention",
    "LRPMultiheadAttentionModule",
    "LRPTransformerEncoderLayer",
    "LRPTransformerEncoder",
    "swap_transformer_layers",
    # attention pooling (Phase 2c)
    "LRPAttentionPooling",
    "LRPAttentionLayer",
    # whole-model wiring (Phase 2c)
    "ModelLRP",
    "prepare_model_for_lrp",
]
