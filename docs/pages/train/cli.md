# Command-Line-Interface of the Train Module

PhysioEx provides a fast and customizable way to train, evaluate and save state-of-the-art models for different physiological signal analysis tasks with different physiological signal datasets. This functionality is provided by the `train`, `test_model` and `finetune` commands provided by this repository.

## Unified dataset flags

All three commands (`train`, `finetune`, `test_model`) share the **same raw-EDF
data layer** and the same dataset flags, so a model trained with a given spec can
be fine-tuned and evaluated with the identical spec:

| Canonical flag | Legacy alias | Meaning |
|---|---|---|
| `--dataset` | `--datasets` | one or more dataset names (e.g. `hmc sleepedf`); multiple names are merged via `MultiDataset` |
| `--channels` | `--selected_channels` | channels to load (modality or physical names) |
| `--pipelines` | `--preprocessing` | preset pipeline name (`raw`, `time_domain`, `time_frequency`, `seqsleepnet`, …) |
| `--sequence_length` | `--seqlen` | epoch sequence length `L` (`-1` for full recordings) |

Additional shared flags: `--dataset_root` (override the data root; else
`PHYSIOEX_DATA`), `--dataset_kwargs` (JSON extra constructor kwargs, e.g.
`'{"cohort": 2}'` for MASS), and `--cache_dir`.

The legacy `PhysioExDataset` (preprocessed-array) layer is **deprecated** and no
longer used by any CLI.

Example — train, then evaluate the checkpoint with the same spec:

```bash
train        --model physioex.models.tinysleepnet:TinySleepNet \
             --dataset hmc --channels EEG EOG EMG --pipelines time_domain \
             --sequence_length 21 --checkpoint_path runs/tsn

test_model   --model physioex.models.tinysleepnet:TinySleepNet \
             --ckpt_path runs/tsn/best.pt \
             --dataset hmc --channels EEG EOG EMG --pipelines time_domain \
             --sequence_length 21 --voting
```

---

`train` CLI
::: bin.train
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4


---

`test` CLI
::: bin.test
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

---

`finetune` CLI
::: bin.finetune
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4