# `SleepPPGNet` (Kotzen et al., 2023) — replica weights

SleepPPG-Net stages a whole night from a single PPG (photoplethysmogram)
channel: a 1-D convolutional encoder over the continuous signal, a window
layer that cuts it into 30-s epochs, a time-distributed dense layer and two
dilated temporal convolution blocks. It predicts four classes (W, Light, Deep,
REM).

- Paper: *SleepPPG-Net: A deep learning algorithm for robust sleep staging
  from continuous photoplethysmography*, IEEE JBHI (2023), arXiv:2202.05735.
- The authors did not release weights. physioex loads a **third-party
  replica** (`sleep-is-all-you-need/SPPPPG-Net` on Hugging Face, retrained with
  the SleePyPhases framework). It is not the authors' model, and its weights
  are licensed for fair non-commercial research use only;
  `load_from_pretrained` warns about this on every load.
- physioex class: `physioex.models.sleepppgnet:SleepPPGNet`.

## Quick start

```python
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train import replication as rep

model = load_from_pretrained("sleepppgnet-replica", verbose=True)

data = get_dataset("mesa")(
    channels=["PPG"],
    pipelines="sleepppgnet",
    sequence_length=0,
    label_transform="aasm4",
    trim_excess_wake=False,
)
test = data.get_splits(split="sleepppgnet-replica")[2]
for subject, signals, labels, metadata in rep.iter_nights(data, test[:1]):
    stages = model.predict_night(signals)   # (n_epochs,) in 0..3
```

The model output is permuted to the physioex class order (W, Light, Deep,
REM); `class_order="original"` keeps the replica's order.

## Inputs and preprocessing

The `sleepppgnet` preset follows the replica's preprocessing: an 8th-order
Chebyshev II low-pass at 8 Hz (-40 dB stop band), FFT resampling to 1024
samples per 30-s epoch (34.13 Hz), clipping at three standard deviations and
z-scoring; the night is padded or truncated to 1200 epochs (10 h).

## Replication

`examples/pretrained/sleepppgnet/verify.py` evaluates the replica on the 204
MESA test nights of its split (`get_splits(split="sleepppgnet-replica")`) with
the paper's metric, the median per-recording kappa. It obtains 0.739 against
the paper's 0.75. Since the weights are not the authors', the number is
reported, not asserted.

## Training

physioex does not ship a training script for this model. The paper's recipe is
summarised in `examples/pretrained/sleepppgnet/TRAINING.md`.

## API

```{eval-rst}
.. autoclass:: physioex.models.sleepppgnet.SleepPPGNet
   :members: predict_night
   :no-index:
```
