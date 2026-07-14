---
orphan: true
---

# `physioex.explain` — class diagram

The explainability toolkit: **post-hoc attribution** (time / frequency / time-
frequency), **foundational** concept analysis (CSD), and **prototype** discovery.
Source: `physioex/explain/`.

## Post-hoc attribution

```mermaid
classDiagram
    class Module { <<torch.nn>> }
    class Saliency {
        +__init__(f, target, ...)
        +forward(x)
    }
    class InputXGradient
    class IntegratedGradients {
        +forward(x, baseline, steps)
    }
    class ExpectedGradients {
        +forward(x, n_samples)
        +set_baselines(baselines)
    }
    class SpectralGradients {
        +__init__(f, fs, freq_step, steps, ...)
        +band_frequencies(len)
        +forward(x)
    }
    Module <|-- Saliency
    Saliency <|-- InputXGradient
    Saliency <|-- IntegratedGradients
    Saliency <|-- ExpectedGradients
    Module <|-- SpectralGradients
    Saliency <|-- FreqSpectralGradients
```

Frequency- and time-frequency-domain variants wrap the same four gradient
methods around differentiable DFT/STFT layers:

```mermaid
classDiagram
    class DFTLayer
    class IDFTLayer
    class STFTLayer
    class ISTFTLayer
    Saliency <|-- DFTSaliency
    InputXGradient <|-- DFTInputXGradient
    IntegratedGradients <|-- DFTIntegratedGradients
    ExpectedGradients <|-- DFTExpectedGradients
    Saliency <|-- STFTSaliency
    InputXGradient <|-- STFTInputXGradient
    IntegratedGradients <|-- STFTIntegratedGradients
    ExpectedGradients <|-- STFTExpectedGradients
    DFTSaliency ..> DFTLayer
    STFTSaliency ..> STFTLayer
```

Support: `functionizer.py` (`Funct`, `SeqFunct` — wrap a classifier into a scalar
target function), `filters.py` (`filtfilt`, `lowpass_filter`, `highpass_filter`),
`metrics/` (`complexity`, `localization`, `infidelity`, `tfle`, `tf_concentration`,
`resolution_product`).

## Foundational (Conceptual Spectral Decomposition)

```mermaid
classDiagram
    class Module { <<torch.nn>> }
    class ConceptualSpectralDecomposition {
        +__init__(model, probe_weights, fs, bands, specificity, ...)
        +explain(x, target_class, embeddings, labels)
    }
    class MultiChannelSpectralGradients {
        +forward(x, skip_ig)
    }
    class CSDResult {
        +concepts
        +class_attribution
        +top_band_hz(dim)
    }
    class ConceptAttribution {
        +weighted_attribution
        +band_energy
        +top_band_idx
    }
    class SpecificityStrategy {
        <<strategy>>
        +str name
        +compute_mask(W, embeddings, labels, x, target_class)
    }
    Module <|-- ConceptualSpectralDecomposition
    Module <|-- MultiChannelSpectralGradients
    ConceptualSpectralDecomposition ..> CSDResult : returns
    CSDResult o-- ConceptAttribution
    ConceptualSpectralDecomposition ..> SpecificityStrategy
    SpecificityStrategy <|-- CohenDSpecificity
    SpecificityStrategy <|-- MarginSpecificity
    SpecificityStrategy <|-- SoftmaxSpecificity
    SpecificityStrategy <|-- TopKSpecificity
    SpecificityStrategy <|-- NoFilter
```

Support: `sleep_bands.py` (`FrequencyBand`, `bands_to_bin_ranges`,
`band_center_frequencies`, `band_names`), `report.py` (`spectral_class_profile`,
`concept_atlas`).

## Prototypes

```mermaid
classDiagram
    class IntegratedGradients
    class PrototypeRelevance {
        +__init__(model, index, steps, ...)
        +forward(x, index, baseline, steps)
    }
    class VQBottleneck {
        +__init__(codebook_init, commitment_weight)
        +forward(z)
    }
    IntegratedGradients <|-- PrototypeRelevance
    Module <|-- VQBottleneck
```

Module-level: `local.py` (`proj_fn`, `get_prototypes`), `reconstruct.py`
(`data_driven_reconstruction`, `model_driven_reconstructions`), `posthoc/nmf.py`
(`discover_prototypes_nmf`), `posthoc/vq.py` (`learn_codebook_kmeans`,
`quantize_embeddings`, `train_codebook`), `posthoc/utils.py`
(`load_epoch_embeddings*`, `nearest_prototype_classify`, `evaluate_metrics`).

## Class reference (responsibilities)

- **`Saliency` family** — Captum-style attributors over a scalar target function `f` (built with `Funct`/`SeqFunct`); `IntegratedGradients`/`ExpectedGradients` add baselines/paths.
- **DFT/STFT variants** — attribute relevance in frequency / time-frequency space via differentiable (I)DFT/(I)STFT layers, subclassing the corresponding time-domain method.
- **`SpectralGradients`** — path-integral attribution across frequency bands.
- **`ConceptualSpectralDecomposition`** — decomposes foundation-model embeddings into interpretable spectral concepts; a pluggable `SpecificityStrategy` selects the concepts most specific to a class; returns a `CSDResult` (list of `ConceptAttribution`).
- **`PrototypeRelevance` / `VQBottleneck`** — prototype relevance (IG-based) and vector-quantized codebook prototypes for concept-level explanation.