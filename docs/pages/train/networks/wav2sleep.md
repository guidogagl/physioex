# `Wav2Sleep` (Carter & Tarassenko, 2024)

wav2sleep stages sleep from any subset of cardiorespiratory signals (ECG, PPG,
abdominal and thoracic effort) recorded over a whole night. Each signal has its
own encoder; per-epoch embeddings are mixed across the available signals with
attention and then across epochs with a dilated CNN. A signal that is missing
is masked, so one model serves every combination of inputs it was trained on.

- Paper: *wav2sleep: A Unified Multi-Modal Approach to Sleep Stage
  Classification from Physiological Signals*, ML4H 2024, arXiv:2411.04644.
- Authors' code and weights: `joncarter1/wav2sleep` (GitHub),
  `joncarter/wav2sleep` and `joncarter/wav2sleep-eog` (Hugging Face), MIT.
- physioex class: `physioex.models.wav2sleep:Wav2Sleep` (a clean
  re-implementation; the original checkpoints load with `strict=True` after
  key conversion).

## Quick start

```python
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train import replication as rep

model = load_from_pretrained("wav2sleep", verbose=True)   # downloads from the Hub

data = get_dataset("mesa")(
    channels=["ECG", "PPG", "ABD", "THX"],
    pipelines="wav2sleep",          # the authors' preprocessing
    sequence_length=0,              # one item = one whole night
    label_transform="aasm4",        # W, Light (N1+N2), Deep, REM
    trim_excess_wake=False,
)
test = data.get_splits(split="wav2sleep")[2]
for subject, signals, labels, metadata in rep.iter_nights(data, test[:1]):
    stages = model.predict_night(signals)   # (n_epochs,) int64 in 0..3
```

`predict_night` accepts any subset of the keys `ECG`, `PPG`, `ABD`, `THX`
(`EOG-L`, `EOG-R` for `wav2sleep-eog`); keys it does not know are ignored.
Nights longer than 10 h are truncated to 1200 epochs.

## Inputs and preprocessing

The `wav2sleep` preset reproduces the authors' ingestion: ECG and EOG are
converted to millivolts, PPG and respiratory effort are scaled by their
physical range; each signal is linearly interpolated onto a grid of 1024
samples per 30-s epoch (256 for ABD/THX, 4096 for EOG), padded with zeros or
truncated to 10 h, and z-scored over the whole grid. The checkpoints use four
classes (`label_transform="aasm4"`).

| name | signals | classes |
|---|---|---|
| `wav2sleep` | ECG, PPG, ABD, THX | 4 (W, Light, Deep, REM) |
| `wav2sleep-eog` | EOG-L, EOG-R | 5 (W, N1, N2, N3, REM) |

## Replication

`examples/pretrained/wav2sleep/verify.py` evaluates the checkpoint on the
paper's test lists (`get_splits(split="wav2sleep")`, all visits of SHHS, MROS
and WSC) with the paper's metric, pooled Cohen's kappa, and compares with
Table 6. As in the authors' evaluation, test nights without any N1, N3 or REM
epoch are left out, and a flat PPG is treated as absent.

The labels and the preprocessed ECG were checked against the authors' own
pipeline: the labels match on every test night of SHHS, MROS, MESA and CCSHS,
and the ECG input matches (r = 1.0000, per-night kappa within 0.0004). With the
public checkpoint, pooled kappa minus Table 6 is:

| dataset | all | ECG+THX | ECG | PPG |
|---|---|---|---|---|
| SHHS | -0.007 | -0.008 | -0.007 | - |
| MESA | -0.006 | -0.019 | -0.021 | -0.011 |
| WSC | -0.003 | -0.003 | -0.011 | - |
| CCSHS | +0.015 | +0.014 | +0.021 | - |
| MROS | -0.005 | -0.006 | -0.005 | - |

The checkpoint on the Hub was uploaded in January 2026, after the paper, and
its model card reports no kappa; the targets are therefore reported, not
asserted. CFS and CHAT (and the census rows that need them) are skipped when
their data are not available.

## Training

physioex does not ship a training script for this model. The authors' recipe
is summarised in `examples/pretrained/wav2sleep/TRAINING.md`.

## API

```{eval-rst}
.. autoclass:: physioex.models.wav2sleep.Wav2Sleep
   :members: predict_night
   :no-index:
```
