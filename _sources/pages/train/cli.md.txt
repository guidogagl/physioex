# Command-line interface of the train module

PhysioEx installs three console scripts for training, fine-tuning and evaluating
models on physiological-signal datasets:

| Command | Entry point |
|---|---|
| `train` | `physioex.train.bin.train:train_script` |
| `finetune` | `physioex.train.bin.finetune:finetune_script` |
| `test_model` | `physioex.train.bin.test:test_script` |

All three build models via a `--model 'module.path:ClassName'` spec and share the
**same raw-EDF data layer**, so a model trained with a given spec can be
fine-tuned and evaluated with the identical spec.

## Unified dataset flags

The dataset flags are shared by all three commands (added by
`physioex.train.bin._common.add_dataset_cli_args`). Legacy aliases write the
same destination, so either spelling works:

| Canonical flag | Legacy alias | Default | Meaning |
|---|---|---|---|
| `--dataset` | `--datasets` | *(required)* | one or more dataset names (e.g. `hmc sleepedf`); multiple names are merged via `MultiDataset` |
| `--channels` | `--selected_channels` | `EEG EOG EMG` | channels to load (modality or physical names) |
| `--pipelines` | `--preprocessing` | `time_domain` | preset pipeline name (`raw`, `time_domain`, `time_frequency`, `seqsleepnet`, `eeg`, `emg`, …) |
| `--sequence_length` | `--seqlen` | `21` | epoch sequence length `L` (`-1` for full recordings) |

Additional shared flags: `--dataset_root` (override the data root; else
`PHYSIOEX_DATA`), `--dataset_kwargs` (JSON/YAML extra constructor kwargs, e.g.
`'{"cohort": 2}'` for MASS), and `--cache_dir`.

The legacy preprocessed-array `PhysioExDataset` layer is **deprecated** and no
longer used by any CLI.

## `train`

Trains a model from scratch and writes checkpoints to `--checkpoint_path`.

Model / config flags: `--model` (required, class spec), `--model_kwargs` (JSON/YAML
constructor kwargs; `in_chan` is auto-injected from the channel count if not
given), `--config` (optional YAML file that overlays the CLI args).

Training flags: `--max_epochs` (20), `--lr` (1e-3), `--weight_decay` (1e-5),
`--train_batch_size` (32), `--eval_batch_size` (1), `--fold` (0),
`--checkpoint_path`, `--gpu_id`, `--num_workers` (0), `--seed` (42),
`--accumulate_grad_batches` (1), `--early_stopping_patience`. Experiment-tracking
flags (TensorBoard / W&B) are added by the logger.

```bash
train --model physioex.models.tinysleepnet:TinySleepNet \
      --dataset hmc --channels EEG EOG EMG --pipelines time_domain \
      --sequence_length 21 --max_epochs 20 --checkpoint_path runs/tsn
```

## `finetune`

Continues training from a pretrained checkpoint (defaults favour a low learning
rate). Requires `--model` and `--ckpt_path` (the checkpoint to start from), plus
`--model_kwargs` and the shared dataset flags. Training flags mirror `train`
with fine-tuning defaults: `--max_epochs` (5), `--lr` (1e-5), `--weight_decay`
(1e-6), `--train_batch_size` (32), `--eval_batch_size` (1), `--fold` (0),
`--checkpoint_path`, `--gpu_id`, `--num_workers` (0).

```bash
finetune --model physioex.models.tinysleepnet:TinySleepNet \
         --ckpt_path runs/tsn/best.pt \
         --dataset dcsm --channels EEG EOG EMG --pipelines time_domain \
         --sequence_length 21 --lr 1e-5 --checkpoint_path runs/tsn_ft
```

## `test_model`

Evaluates a checkpoint. Requires `--model` and `--ckpt_path`, plus the shared
dataset flags. Other flags: `--config`, `--fold` (0), `--results_path`
(directory to save the results CSV), `--gpu_id`, and `--voting` (enable
sliding-window per-subject voting, recommended for full-night sequences).

```bash
test_model --model physioex.models.tinysleepnet:TinySleepNet \
           --ckpt_path runs/tsn/best.pt \
           --dataset hmc --channels EEG EOG EMG --pipelines time_domain \
           --sequence_length 21 --voting --results_path runs/tsn
```

:::{tip}
Run any command with `--help` for the authoritative, always-current flag list.
The underlying Python API is documented on the [Training](train.md) page.
:::
