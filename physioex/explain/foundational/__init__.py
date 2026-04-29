"""Explainability methods for foundation model embeddings.

Available methods:
    - CSD (Conceptual Spectral Decomposition): time-frequency attribution
      of class-specific embedding dimensions via masked SpectralGradients.
"""
from physioex.explain.foundational.csd import (
    ConceptualSpectralDecomposition,
    CSDResult,
    ConceptAttribution,
)
from physioex.explain.foundational.sleep_bands import (
    FrequencyBand,
    SLEEP_BANDS,
    SLEEP_BAND_NAMES,
)
from physioex.explain.foundational.specificity import (
    SpecificityStrategy,
    CohenDSpecificity,
    MarginSpecificity,
    SoftmaxSpecificity,
    TopKSpecificity,
    NoFilter,
)

__all__ = [
    "ConceptualSpectralDecomposition",
    "CSDResult",
    "SpecificityStrategy",
    "CohenDSpecificity",
    "MarginSpecificity",
    "SoftmaxSpecificity",
    "TopKSpecificity",
    "NoFilter",
]
