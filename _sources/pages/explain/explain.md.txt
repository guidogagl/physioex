# Explain Module

PhysioEx ships an explainability (XAI) toolkit for sleep-staging and
foundation models, organized in three families under `physioex.explain`. For
the full class diagram see the
[explain-layer architecture](../architecture/library/explain.md) page.

## Post-hoc attribution — `physioex.explain.posthoc`

Gradient-based attribution over a trained classifier, plus frequency- and
time-frequency-domain variants:

- **Time-domain**: `Saliency`, `InputXGradient`, `IntegratedGradients`,
  `ExpectedGradients` (`physioex/explain/posthoc/gradients.py`). Each is a
  `torch.nn.Module` built around a scalar-valued target function `f` (wrap a
  classifier into one with `Funct` / `SeqFunct` from
  `physioex/explain/posthoc/functionizer.py`); attribution is produced by
  calling the attributor on the input.
- **Spectral**: `SpectralGradients` (`spectralgradients.py`) — path-integral
  attribution across frequency bands.
- **Frequency-resolved**: DFT- and STFT-domain attributors, exported from
  `vidft.py` and `vistdft.py`, wrap the same gradient methods around
  differentiable (I)DFT/(I)STFT layers.
- **Faithfulness metrics** under `posthoc/metrics/` quantify attribution quality
  (complexity, localization, infidelity, and time-frequency measures).

Refer to the [API Reference](../../api/index.md) for the exact constructor and
call signatures of each attributor before use.

## Foundational explainability — `physioex.explain.foundational`

Concept-level analysis of foundation-model embeddings:

- **Conceptual Spectral Decomposition** — `ConceptualSpectralDecomposition`
  (`csd.py`) decomposes embeddings over interpretable spectral concepts and
  returns a `CSDResult` (a list of `ConceptAttribution`).
- **Specificity strategies** — a pluggable `SpecificityStrategy`
  (`CohenDSpecificity`, `MarginSpecificity`, `SoftmaxSpecificity`,
  `TopKSpecificity`, `NoFilter`) selects the concepts most specific to a class.
- **Sleep bands** (`sleep_bands.py`), **multi-channel spectral gradients**
  (`multichannel_sg.py`), and reporting helpers (`report.py`).

## Prototypes — `physioex.explain.prototypes`

Prototype/concept discovery and reconstruction:

- **Local relevance**: `PrototypeRelevance`, `get_prototypes` (`local.py`).
- **NMF** prototype discovery (`posthoc/nmf.py`:
  `discover_prototypes_nmf`).
- **Vector-quantized** codebooks: `VQBottleneck` and the codebook helpers in
  `posthoc/vq.py` (`learn_codebook_kmeans`, `quantize_embeddings`,
  `train_codebook`).
- **Reconstruction** of learned concepts (`reconstruct.py`).

See the [API Reference](../../api/index.md) for verified signatures across all
three families.
