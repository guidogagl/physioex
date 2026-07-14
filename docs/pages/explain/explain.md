# Explain Module

PhysioEx ships a rich explainability (XAI) toolkit for sleep-staging and
foundation models, organized in three families under `physioex.explain`.

## Post-hoc attribution — `physioex.explain.posthoc`

Gradient-based attribution methods wrapping [Captum](https://captum.ai), plus
frequency-domain variants, for any trained PhysioEx classifier:

- **Time-domain**: `Saliency`, `InputXGradient`, `IntegratedGradients`,
  `ExpectedGradients` (`physioex/explain/posthoc/gradients.py`).
- **Spectral**: `SpectralGradients` (`spectralgradients.py`) attributes relevance
  across frequency bands.
- **Frequency-resolved**: `ViDFT` (`vidft.py`) and `VisTDFT` (`vistdft.py`)
  produce DFT/STFT-domain attributions.
- **Faithfulness metrics** under `posthoc/metrics/` quantify attribution quality.

```python
from physioex.explain.posthoc.gradients import IntegratedGradients

attributor = IntegratedGradients(model)
relevance = attributor.attribute(inputs)   # same shape as inputs
```

## Foundational explainability — `physioex.explain.foundational`

Concept-level analysis of foundation-model embeddings:

- **Conceptual Spectral Decomposition** — `ConceptualSpectralDecomposition`
  (`csd.py`) decomposes embeddings over interpretable spectral concepts.
- **Sleep bands** (`sleep_bands.py`) and **specificity strategies**
  (`CohenDSpecificity`, `MarginSpecificity`, `SoftmaxSpecificity`,
  `TopKSpecificity`) select the concepts most specific to a class.
- **Multi-channel spectral gradients** (`multichannel_sg.py`).

## Prototypes — `physioex.explain.prototypes`

Prototype/concept discovery and reconstruction:

- **Local relevance**: `PrototypeRelevance`, `get_prototypes` (`local.py`).
- **NMF** prototype discovery (`posthoc/nmf.py`).
- **Vector-quantized** codebooks: `VQBottleneck` (`posthoc/vq.py`).
- **Reconstruction** of learned concepts (`reconstruct.py`).

See the runnable scripts under `examples/explain/` for end-to-end usage.
