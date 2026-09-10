"""PhysioEx explainability subsystem.

Sub-packages:
    - ``posthoc``    — gradient-based attribution (Saliency, InputXGradient,
      Integrated / Expected Gradients, and their DFT/STFT spectral variants).
    - ``foundational`` — spectral-concept methods (CSD, specificity, ...).
    - ``prototypes`` — prototype / concept discovery and reconstruction.
    - ``lrp``        — Layer-wise Relevance Propagation (needs the ``explain``
      extra: ``pip install "physioex[explain]"``).

``LRP`` is re-exported lazily so that ``import physioex.explain`` does not pull
in the optional ``zennit`` / ``lxt`` dependencies unless LRP is actually used.
"""

__all__ = ["LRP"]


def __getattr__(name):
    # PEP 562 lazy attribute — only import the LRP stack on first access.
    if name == "LRP":
        from physioex.explain.lrp import LRP

        return LRP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
