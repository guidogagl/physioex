# `JonesECGNet` (Jones, Itti & Sheth, 2024)

An ECG-only sleep staging network: a convolutional front-end turns each 30-s
epoch of single-lead ECG at 256 Hz into features, dense residual blocks add
age, sex and the time since midnight, and a temporal convolutional network
classifies the whole night into the five AASM stages.

- Paper: *Expert-level sleep staging using an electrocardiography-only feed-forward
  neural network*, Computers in Biology and Medicine (2024),
  doi:10.1016/j.compbiomed.2024.108545.
- Authors' code and weights: `adammj/ecg-sleep-staging` (GitHub), AGPL-3.0.
- physioex class: `physioex.models.jones_ecg:JonesECGNet`, written from the
  paper and the checkpoint layout (no code copied from the AGPL repository).

## Quick start

```python
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train import replication as rep

model = load_from_pretrained("jones-ecg", verbose=True)   # = "jones-ecg-primary"

data = get_dataset("mesa")(
    channels=["ECG"],
    pipelines="jones_ecg",        # port of the authors' MATLAB preprocessing
    sequence_length=0,            # one item = one whole night
    trim_excess_wake=False,       # 5 AASM classes, no label transform
)
test = data.get_splits(split="wav2sleep-census")[2]
for subject, signals, labels, metadata in rep.iter_nights(data, test[:1]):
    # metadata: {"age": years, "sex": "M"/"F", "recording_start": datetime.time}
    stages = model.predict_night(signals, metadata)   # (n_epochs,) in 0..4
```

When age, sex or the recording start are missing, `predict_night` fills the
corresponding inputs as the variant does for inputs it was not trained with
(standard-normal noise) and logs a warning once.

## Inputs and preprocessing

The `jones_ecg` preset applies `JonesECGPreprocess` to the ECG channel: removal
of loose-lead clipping, a zero-phase Chebyshev II high-pass (0.5 Hz), a notch
at the mains frequency plus an iterative search for further narrow-band noise,
resampling to 256 Hz, R-peak detection with the authors' templates
(`archetypes.mat`, downloaded on first use) and scaling by twice the 90th
percentile of the R-wave amplitude, clipped to [-1, 1]. The module docstring
lists where the port deviates from the MATLAB code. NSRR readers provide the
subject metadata through `subject_metadata(subject_id)`.

Seven checkpoints are available, all with the same hyper-parameters:

| name | metadata used | temporal network |
|---|---|---|
| `jones-ecg-primary` (`jones-ecg`) | age, sex, start time | symmetric |
| `jones-ecg-realtime` | age, sex, start time | causal |
| `jones-ecg-no-demographics` | start time | symmetric |
| `jones-ecg-no-time` | age, sex | symmetric |
| `jones-ecg-only-ecg` | none | symmetric |
| `jones-ecg-cyclic` | age, sex, cyclic start time | symmetric |
| `jones-ecg-cyclic-realtime` | age, sex, cyclic start time | causal |

## Replication

`examples/pretrained/jones-ecg/verify.py` evaluates a variant on the
census-balanced held-out set (`get_splits(split="wav2sleep-census")`) with the
paper's metric, the median of the per-recording kappas, and prints next to it
the authors' own per-recording kappas on exactly the same recordings. Results
of the primary model:

| dataset | paper | physioex | status |
|---|---|---|---|
| WSC | 0.6141 | 0.6141 | PASS |
| MESA | 0.6750 | 0.6714 | PASS |
| CCSHS | 0.8136 | 0.8034 | outside tolerance |

On CCSHS (60 nights) the median per-recording difference from the authors'
kappas is -0.001, but single recordings differ by up to 0.13: the
preprocessing port is not bit-exact. CFS and CHAT, and the census rows that
need them, are skipped when their data are not available.

## Training

physioex does not ship a training script for this model. The authors' recipe
is summarised in `examples/pretrained/jones-ecg/TRAINING.md`.

## API

```{eval-rst}
.. autoclass:: physioex.models.jones_ecg.JonesECGNet
   :members: predict_night, build_additional, midnight_offset
   :no-index:

.. autoclass:: physioex.data.steps.JonesECGPreprocess
   :no-index:
```
