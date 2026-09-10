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

__all__ = [
    "LRP",
    "physioex_composite",
    "epsilon_composite",
    "default_canonizers",
]
