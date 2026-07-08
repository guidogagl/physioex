# Copilot instructions for `physioex`

## Build, test, and lint commands

```bash
# install (editable)
pip install -e .

# include dev tools used in this repo
pip install -e ".[dev]"

# lint/format check (CI uses Black on ./physioex)
black --check --verbose ./physioex
```

Tests in this repository are mostly integration-style module scripts under `tests/` (many are not pytest-style unit tests).

```bash
# run a full test module
python -m tests.test_pipeline
python -m tests.test_cache
python -m tests.test_trainer_integration

# run one unittest test method (for unittest-based files)
python -m unittest tests.test_cli_workflows.TestTrainCLI.test_help_exits_zero
```

Several dataset-specific tests require local dataset files and credentials/paths to be available (for example `test_phase_*`, `test_mass_dataset`, `test_wsc_dataset`, `test_alzheimers_parkinsons`).

## High-level architecture

PhysioEx has two dataset/training paths that coexist:

1. **New raw EDF path (`physioex.data.base.BasePhysioDataset`)**  
   Lazy reads EDF channels, applies composable preprocessing pipelines, and caches per-channel outputs on disk (`ChannelCache`).
2. **Legacy preprocessed path (`physioex.data.dataset.PhysioExDataset`)**  
   Kept for backward compatibility with older workflows/checkpoints.

The modern flow is:

`Dataset registry (name -> class) -> channel resolution -> preprocessing preset/pipeline -> on-disk cache -> Trainer -> evaluation/explainability`

Important wiring:

- `physioex/data/datasets/__init__.py`: dataset registry via `get_dataset(name)`.
- `physioex/data/presets.py`: preset registry; returns either one pipeline for all channels or per-modality pipeline bundles.
- `physioex/data/collate.py`: dict-batch collation and `stack_channels` to produce `(B, L, C, ...)` tensors.
- `physioex/train/trainer.py`: accepts both dict-style batches (new path) and tuple-style batches (legacy path), and evaluates raw datasets at subject level using full recordings.
- `physioex/models/foundation/`: frozen pretrained encoders + trainable heads; names resolved through `FOUNDATION_MODELS`.

CLI behavior currently mixes both paths:

- `train` uses the **new raw EDF dataset path** (`get_dataset(...)` + `pipelines` + `channels`).
- `finetune` and `test_model` use the **legacy `PhysioExDataset` path**.

## Key conventions for this codebase

- **Model selection is string-based**: CLI expects `--model module.path:ClassName`, dynamically imported at runtime.
- **`model_kwargs` is string-encoded config**: parse JSON first, then YAML fallback.
- **YAML config overlay is supported in CLI**: `--config file.yaml` overrides parsed CLI args for matching keys.
- **Label contract is fixed AASM mapping**: `W=0, N1=1, N2=2, N3=3, REM=4`, and `-1` means unscored/ignored (`ignore_index=-1` in training/eval paths).
- **Dict-batch contract is strict in new data path**: batches carry `signals`, `channel_order`, `labels` (plus metadata). `channel_order` must match within a batch.
- **Caching and loader defaults are intentional**: for `BasePhysioDataset`, `num_workers` defaults to `0` to avoid NFS cache contention.
- **Checkpoint format convention**: saved checkpoints use `{"model_state_dict", "optimizer_state_dict", "epoch"}` in `.pt` files; loader retains compatibility with older formats.
