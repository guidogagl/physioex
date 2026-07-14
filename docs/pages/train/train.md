# Training (programmatic API)

`physioex.train` is a **hand-written PyTorch training/evaluation loop** (not
PyTorch Lightning). The programmatic entry point is `physioex.train.trainer.Trainer`,
a stateless helper exposing only static/class methods; models are plain
`torch.nn.Module`s.

For most workflows the end-user entry points are the command-line tools
(`train`, `finetune`, `test_model`) documented on the [CLI page](cli.md). This
page covers the underlying Python API. For the full class diagram see the
[train-layer architecture](../architecture/library/train.md) page.

## `Trainer` at a glance

`Trainer` methods accept any dataset exposing `.split(fold)` and
`.sequence_length` — i.e. a `BasePhysioDataset` or a `MultiDataset` — and wire
`dict_collate_fn` automatically. Key methods:

- **`build_dataloaders(dataset, train_batch_size=32, eval_batch_size=1, ..., fold=0)`**
  — returns `(train_loader, val_loader)`.
- **`train(model, dataset, checkpoint_path=None, max_epochs=10, lr=1e-3, weight_decay=1e-5, ...)`**
  — runs the epoch loop (AMP, gradient accumulation, optional early stopping,
  pluggable logger) and returns the trained `model`.
- **`evaluate(model, dataset, metrics=..., ...)`** — standard evaluation,
  returns a metrics dict.
- **`voting_evaluate(model, dataset, L=21, ...)`** — sliding-window per-subject
  voting evaluation over full-night sequences.
- **`save_checkpoint(model, path, epoch=None, optimizer=None)`** /
  **`load_checkpoint(model, path, optimizer=None)`** — dual-format
  (`.pt` / Lightning `.ckpt`) checkpoint I/O.

The module-level `physioex.train.trainer.seed_everything(seed=42)` seeds Python,
NumPy and PyTorch RNGs.

## Example

The `Trainer` methods are static/class methods, so you call them directly on the
class — no instance is created:

```python
import torch
from physioex.data.datasets import get_dataset
from physioex.train.trainer import Trainer, seed_everything
from physioex.models.tinysleepnet import TinySleepNet

seed_everything(42)

# 1. Build a dataset (raw-EDF, lazy) — see the Data and Preprocessing pages.
dataset = get_dataset("hmc")(
    channels=["EEG"], pipelines="time_domain", sequence_length=21,
)

# 2. Build a model (a plain nn.Module). in_chan matches the channel count.
model = TinySleepNet(n_classes=5, in_chan=1)

# 3. Train. Trainer builds the dataloaders internally from the dataset.
model = Trainer.train(
    model=model,
    dataset=dataset,
    checkpoint_path="runs/tsn",
    max_epochs=20,
    lr=1e-3,
    fold=0,
)

# 4. Evaluate with sliding-window voting.
metrics = Trainer.voting_evaluate(model=model, dataset=dataset, L=21, fold=0)
print(metrics)
```

:::{note}
The exact constructor arguments differ per model (`n_classes`, `in_chan`, and
model-specific hyper-parameters). Consult each architecture's signature in the
[API Reference](../../api/index.md) or the
[models architecture](../architecture/library/models.md) page before
instantiating.
:::

Fine-tuning is the same `train` call starting from a loaded checkpoint (a lower
learning rate is typical). Multi-GPU DDP training is available via
`physioex.train.multidevicetrainer.MultiDeviceTrainer`, which subclasses
`Trainer` on the same data path.

## API

```{eval-rst}
.. autoclass:: physioex.train.trainer.Trainer
   :members: build_dataloaders, train, evaluate, voting_evaluate,
             save_checkpoint, load_checkpoint
   :show-inheritance:

.. autofunction:: physioex.train.trainer.seed_everything
```

See the full auto-generated [API Reference](../../api/index.md) for metrics,
statistics and logging helpers.
