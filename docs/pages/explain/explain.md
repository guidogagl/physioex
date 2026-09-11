# Explain Module

PhysioEx ships an explainability (XAI) toolkit for sleep-staging and
foundation models, organized under `physioex.explain`: three core families
(post-hoc gradients, foundational/concept, prototypes) plus an optional
**Layer-wise Relevance Propagation** family (`explain` extra). For
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

## Layer-wise Relevance Propagation — `physioex.explain.lrp`

Conservation-based attribution (Bach et al. 2015): relevance is redistributed
backward through the network, conserving the total (`Σ R ≈ f(x)`). This family
is **optional** — install the extra:

```bash
pip install "physioex[explain]"
```

Two entry points, both seeding the target neuron with its **logit value** so
that `Σ R ≈ f_c(x)` and returning relevance shaped like the input:

| Model family | Entry point | Rules |
|---|---|---|
| Feed-forward / CNN (Tsinalis, Chambon2018) | `LRP` (Zennit) | ε dense · γ conv · **w²** first layer (unbounded EEG, *not* the pixel z-box) · BatchNorm canonized · αβ/flat/z-box configurable via `physioex_composite` |
| Recurrent (TinySleepNet, SeqSleepNet, L-SeqSleepNet, ProtoSleepNet-seq) | `ModelLRP` | **Arras signal-take** LSTM/GRU (`LRPLSTM`/`LRPGRU`, cell-level, forward identical to the fused module) · ε on the learnable filterbank / linears · CP-LRP on the attention poolings |
| Attention / transformer (SleepTransformer, CoReSleep, ProtoSleepNet-tf) | `ModelLRP` | **CP-LRP** attention (value path; Ali et al. 2022) · identity rule on LayerNorm/GELU · proportional rule on residuals · ε on projections/FFN |

```python
from physioex.explain.lrp import LRP, ModelLRP

relevance = LRP(cnn_model, out_index=2)(x)                       # Zennit path
relevance, report = ModelLRP(seq_model, in_index=10, out_index=2)(x, return_report=True)
print(report)   # Σ R / f per sample — 1.0 = exact conservation
relevance = ModelLRP(coresleep, output_key="combined")(x)      # dict outputs
```

**How `ModelLRP` works.** The trained model is deep-copied and rewritten
(`prepare_model_for_lrp`): fused `nn.LSTM`/`nn.GRU` become `LRPLSTM`/`LRPGRU`
(subclasses, so `isinstance` checks and the `(output, states)` interface keep
working), `nn.TransformerEncoder`/`MultiheadAttention` become their CP-LRP
versions, PhysioEx's softmax poolings (`AttentionPooling`, `AttentionLayer`,
`ChannelMixer`) and the `LearnableFilterbank` get dedicated adapters, remaining
`Linear`/`Conv` leaves get ε-LRP, BatchNorm is folded into the preceding layer,
and norms/activations use the identity rule. Plain `+` residuals written in a
model's own `forward` are redirected at runtime to the proportional rule
(`patch_residuals=True`) — a plain add would give both branches the full
relevance and over-count. Register your own blocks with
`register_lrp_adapter(cls, factory)`; `audit_lrp_coverage(model)` (also
`ModelLRP(...).uncovered`, and `strict=True`) lists parametric leaves left on
plain autograd, which would propagate *gradient* instead of relevance.

**Reading the numbers.** Conservation is exact (up to ε) for bias-free
networks; **biases absorb relevance** (LRP-ε convention), so on real models
`Σ R / f` is below 1 — `ConservationReport.absorbed` shows how much. CP-LRP
treats the attention matrix as constant: features acting only through *where
to attend* (query/key path) receive zero relevance by design; full AttnLRP
(Achtibat et al. 2024) is non-conserving and not implemented. All stabilisers
are signed (`z + ε·sign z`), as in Arras et al. Not supported (raise): attention
masks, `batch_first=False` transformers, explicit RNN initial states, `proj_size`;
ProtoSleepNet's VQ path (`quantize=True`) runs under `no_grad` and stops relevance.

- **Composites** (`lrp/composites.py`): `physioex_composite` (ε/γ/w²),
  `epsilon_composite` (pure-ε reference). **Canonizers** (`lrp/canonizers.py`):
  `default_canonizers` (BatchNorm merge).
- **Blocks**: `lrp/recurrent.py`, `lrp/transformer.py`, `lrp/pooling.py`;
  shared primitives in `lrp/_functional.py`; diagnostics in `lrp/diagnostics.py`.
- **Backends**: [Zennit](https://github.com/chr5tphr/zennit) for the composites
  and BatchNorm canonization; the recurrent, attention and pooling rules are
  implemented in PhysioEx (`_functional.py`, `_rules.py`) following Arras et al.,
  Ali et al. (CP-LRP) and the LXT conventions — no LXT dependency.

See the [API Reference](../../api/index.md) for verified signatures across all
families.
