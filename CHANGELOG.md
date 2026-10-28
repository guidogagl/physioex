# Changelog

All notable changes to PhysioEx are documented here. Versions follow the
`physioex.__version__` string (read dynamically by the build) and the `v*` git tags
that publish to PyPI.

## 2.0.5 — 2026-10-28

### Added
- **Pretrained ProtoSleepNet and ProtoSleepTransformer** (multi-dataset, 28 training sites) on the Hub as `protosleepnet-gagliardi` and
  `protosleeptransformer-gagliardi` in `4rooms/physioex`. A documentation page lists their training data, recipe and per-site test
  performance (User Guide → Model architectures), and `examples/pretrained/protosleepnet/export_release.py` builds the Hub folder from a
  training checkpoint and its per-site evaluation without recomputing any metric.

### Changed
- `load_from_pretrained(..., verbose=True)` prints only the metric columns that `metrics.json` provides (and `n_subjects`) instead of
  `0.0000` for missing ones, and widens the first column for long site names.

### Fixed
- The ProtoSleepNet example and the README pointed to a model name and a private repository that external users cannot load.

## 2.0.4 — 2026-10-16

Data-layer fixes found by an audit of what physioex reads against the raw files of every dataset. Label caches of
Sleep-EDF, WSC and Alzheimers are recomputed on first use (`LABELS_REVISION` 2, 2 and 3). Cached WSC events are not
versioned: delete `events/<subject>.json` of the 26 allscore nights whose events changed.

### Fixed
- **WSC `.allscore.txt`:** labels and events are anchored at `START RECORDING`. A `STAGE - NO STAGE` line timed before it
  moved the time origin back by hours and shifted every later label (45 of 743 files). Midnight is a backward jump of
  the clock, `PAUSED` to `RECORDING RESUMED` stretches are removed (the EDF has no samples there), and labels or events
  past the end of the EDF are dropped. A file without `START RECORDING` yields no labels, as before.
- **Sleep-EDF:** the hypnogram always runs to 24 h; its tail is cut at the end of the PSG instead of being paired with
  zero padding (in one night 614 epochs of `Wake`).
- **Alzheimers:** labels past the end of a truncated EDF are dropped.
- **MESA:** `*_Off` electrode-status flags (1 Hz) are classified `OTHER`, not EEG/EOG/EMG.

### Added
- **WSC events from `.sco.txt`** (the 1827 Gamma-system nights; only the 743 Twin-system `.allscore.txt` nights had
  events). Same event types as the allscore parser. `Scan # x2` counts samples at twice the EDF rate; a few files write
  the scan with two extra digits (the clock column is used) and 548 write the length ×100 with a negative sign.

## 2.0.3 — 2026-10-15

CCSHS dataset, opt-in utilities for multi-dataset training, and three data-layer fixes. STAGES and Alzheimers label
caches are recomputed on first use, and event counts per type change (see "`artifact` event type").

### Added — CCSHS dataset (`physioex.data.datasets.ccshs`)

`get_dataset("ccshs")`: Cleveland Children's Sleep and Health Study (NSRR `ccshs-trec`, 515 recordings). EEG
`(C3, A2)` / `(C4, A1)` and EOG `(LOC, ROC)` at 128 Hz, chin EMG at 256 Hz; stage 3 and 4 merged to N3,
Movement/unscored to -1, labels indexed by event onset, unknown stage vocabulary raises. Eleven recordings are
excluded by default (`CCSHSDataset.EXCLUDED_SUBJECTS`, `exclude_subjects=[]` keeps them): nine whose stage minutes
disagree with the NSRR summary table and two without a row in it. Metadata is read from
`datasets/ccshs-trec-dataset-*.csv` whatever its version. Tests in `tests/test_ccshs.py`.

### Added — site-balanced sampling (`physioex.data.sampling`)

`site_balanced_weights` / `make_site_sampler` (also exported from `physioex.data`): sampling weights that give every
site the same mass and, inside a site, every subject the same mass, for training on collections of datasets of very
different sizes with `MultiDataset`. Tests in `tests/test_sampling.py`.

### Added — opt-in channel-accuracy tracking during training (`ChannelMixer`)

`ChannelMixer.track_train_acc` and `acc_momentum` update the per-channel accuracy that drives the accuracy-weighted
modality dropout from the training step too (EMA of the clean per-channel logits). Off by default, so the released
models and existing training runs behave exactly as before. Tests in `tests/test_channel_acc_tracking.py`.

### Fixed — subjects without any requested channel (`BasePhysioDataset`)

A subject for which none of the requested channels can be resolved was returned as a block of zeros with the wrong
shape (`(L,)` instead of `(L, F, T)`), which made `dict_collate_fn` fail on any batch containing it. Its zeros now have
the per-epoch shape of the pipeline output, like the zero-filled channels of subjects that miss only some channels, and
a warning names the subject. Subjects are still never dropped.

### Fixed — stale cached labels (`BasePhysioDataset.LABELS_REVISION`)

Cached labels had no version, so after a fix of a label reader the old caches kept serving the wrong labels without any
warning. Datasets now declare `LABELS_REVISION` (default 1); the revision is stored with the cached labels and a cache
with a lower revision is recomputed (with a warning), or used with a warning when the source files are not available.
Caches without the key count as revision 1, so existing caches of unchanged readers stay valid. `STAGESDataset` and
`AlzheimersDataset` are at revision 2 (timestamp-based alignment of STAGES labels, onset-indexed Alzheimer labels):
caches written before those fixes are recomputed on first use. Cached EDF headers that predate the `start_sec` field
(STAGES labels are aligned to the EDF start time with it) are probed again, or used with a warning when the EDF is not
available.

### Changed — `artifact` event type

"SpO2 artifact" intervals (an unreliable oximeter signal, not a physiological event) were counted as `respiratory` by
the NSRR parser and as `desaturation` by the STAGES parser. They now have their own type `artifact`; the concept text is
unchanged. Event counts per type change accordingly, for example in CCSHS `respiratory` drops from 51 915 to 15 040
and `artifact` has 36 875.

## 2.0.2 — 2026-10-10

### Fixed — accuracy-weighted modality dropout (`physioex.models.protosleepnet`)

`ZeroEmbeddingDropout` now samples the channels to occlude with probability
proportional to their accuracy, `p_c = a_c / sum_j a_j`, so the channels the
model relies on most are dropped most often. This is the rule the released
ProtoSleepNet and ProtoSleepTransformer models were trained with; a later
refactoring had inverted the sampling weights to `1 - a_c`. When no accuracy
estimate is available yet the sampling is uniform. Unit tests in
`tests/test_modality_dropout.py`.

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
