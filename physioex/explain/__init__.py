"""PhysioEx explainability subsystem.

Sub-packages:
    - ``posthoc``      — gradient-based attribution (Saliency, InputXGradient,
      Integrated / Expected Gradients, and their DFT/STFT spectral variants).
    - ``foundational`` — spectral-concept methods (CSD, specificity, ...).
    - ``prototypes``   — prototype / concept discovery and reconstruction.
    - ``lrp``          — Layer-wise Relevance Propagation (needs the ``explain``
      extra: ``pip install "physioex[explain]"``).

The LRP entry points are re-exported lazily so that ``import physioex.explain``
does not pull in the optional ``zennit`` / ``lxt`` dependencies.
"""

_LAZY = {
    "LRP": "physioex.explain.lrp.attributor",
    "ModelLRP": "physioex.explain.lrp.model",
    "prepare_model_for_lrp": "physioex.explain.lrp.model",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    # PEP 562 lazy attribute — only import the LRP stack on first access.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
