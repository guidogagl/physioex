# `Chambon2018`

This page documents the `Chambon2018Net` architecture, published
[here](https://ieeexplore.ieee.org/document/8307462). It encodes each 30-second
epoch independently (via braindecode's `SleepStagerChambon2018`), then
concatenates the per-epoch features across the sequence to classify the central
epoch.

Train it with the CLI:

```bash
train --model physioex.models.chambon2018:Chambon2018Net \
      --dataset hmc --channels EEG EOG EMG --pipelines time_domain
```

```{eval-rst}
.. autoclass:: physioex.models.chambon2018.Chambon2018Net
   :members: __init__, encode, forward
   :show-inheritance:
```
