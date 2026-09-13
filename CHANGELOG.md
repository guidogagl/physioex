# Changelog

All notable changes to PhysioEx are documented here. Versions follow the
`physioex.__version__` string (read dynamically by the build) and the `v*` git tags
that publish to PyPI.

## 2.0.1 — 2026-09-13

### Added — Layer-wise Relevance Propagation (`physioex.explain.lrp`)

A conservation-based attribution family (Bach et al. 2015) covering the whole
model zoo. Install with the new optional extra `pip install "physioex[explain]"`
(Zennit only; no LXT dependency).

- **Two entry points**, both seeding the target neuron with its logit so that
  `Σ R ≈ f_c(x)`, both returning a per-sample `ConservationReport` with
  `return_report=True`:
  - `LRP` — Zennit composites for feed-forward / CNN models (Tsinalis,
    Chambon2018): ε on dense, γ on convolutions, **w²** on the first layer
    (unbounded EEG, not the pixel z-box), BatchNorm canonized; αβ / flat / z-box
    configurable via `physioex_composite`.
  - `ModelLRP` — whole-model LRP for recurrent, attention and transformer
    architectures (TinySleepNet, SeqSleepNet, L-SeqSleepNet, SleepTransformer,
    CoReSleep, ProtoSleepNet): **Arras signal-take** rule for LSTM/GRU at cell
    level (`LRPLSTM`/`LRPGRU`, drop-in subclasses with a bit-identical forward,
    `gate_rule="signal_take" | "uniform"`), **CP-LRP** for attention (value
    path, exact conservation; Ali et al. 2022), identity rule for
    LayerNorm/activations, proportional rule for residual sums, ε for linear
    leaves and PhysioEx's learnable filterbank, CP-LRP adapters for the softmax
    poolings (`AttentionPooling`, `AttentionLayer`, `ChannelMixer`).
- `prepare_model_for_lrp` rewrites a deep copy of the model; plain `+` residuals
  written in a model's own `forward` are redirected to the proportional rule at
  runtime (`patch_residuals`); `audit_lrp_coverage` / `strict=True` report any
  parametric leaf left on plain autograd; `register_lrp_adapter` adds custom
  blocks. Signed, dtype-safe stabiliser `z + ε·sign z` everywhere (fp16-safe).
- Docs: entry-point/rule matrix, how to read conservation and choose ε (ε must be
  scaled to the activations: measured on the pretrained SeqSleepNet, 1e-6 inflates
  relevance 2–3×, 1e-2 conserves), unsupported cases, and a literature note on
  LRP vs path-based attribution (Deep Taylor, Integrated Gradients,
  DeepLIFT/DeepSHAP). `physioex.explain.lrp` is in the API reference.
- Example: `examples/explain/lrp_seqsleepnet_n3.py` explains the N3 logit of the
  pretrained `seqsleepnet-phan` on MASS training sequences with Saliency,
  Input×Gradient, Integrated Gradients and LRP under different rule assignments.
- Tests: 91 new tests (`tests/explain/lrp/`), including forward-equivalence to
  the fused modules, exact conservation on bias-free blocks and models, and all
  six real architectures end-to-end with full rule coverage.

### Known limitations

CP-LRP assigns zero relevance to the attention query/key path by design (full
AttnLRP is not implemented); bias relevance is absorbed (LRP-ε convention);
attention masks, `batch_first=False` transformers, explicit RNN initial states
and `proj_size` raise; ProtoSleepNet's VQ path runs under `no_grad`. Foundation
encoders and frequency-domain (DFT/STFT) LRP are planned follow-ups.

## 2.0.0 — 2026-07-15

Ground-up release: raw-EDF lazy data layer with 12 dataset loaders, 9 pretrained
foundation encoders behind a uniform `encode()` interface, hand-written `Trainer`,
pytest suite and Sphinx documentation. Full notes in the
[v2.0.0 GitHub Release](https://github.com/guidogagl/physioex/releases/tag/v2.0.0).
