# `MSVQ`

`MSVQ` is a self-supervised encoder of single-channel clean EEG that writes each
30-second epoch as three codes from three small dictionaries, one per time scale:

- **N** (context): what the ±20 surrounding epochs say about the epoch, with the epoch itself masked out;
- **E** (epoch): what the epoch adds to N;
- **P** (patch): what each 1-second patch adds to N+E.

Each scale quantizes a residual in the latent space, i.e. its own representation
minus the part already explained by the coarser codes. Encoder, dictionaries and
decoder are trained together, and every prefix of the cascade redraws the epoch:

- N and N+E draw the epoch log-PSD, every frequency bin z-scored so that no band dominates;
- N+E+P draws each 1-second waveform: log-magnitude, phase on the unit circle (NeuroRVQ-style), and the time signal.

Label-free cluster targets (band log-powers normalized per night, k-means) can
be added as an auxiliary loss. Sleep stages are never used. `--no-vq` trains the
same network with continuous latents: it is the upper bound used to choose the
dictionary sizes.

Train encoder and decoder on one dataset, keeping only the main sleep period of
each recording:

```bash
python -m physioex.train.bin.msvq --dataset dcsm --out outputs/msvq_dcsm/run0
```

The dataset option `trim_mode="main_period"` keeps the block with the most
sleep, i.e. sleep separated by gaps of at most 1 h, widened by 30 min on each
side. Labels outside that block are set to `-1`, and windows centred outside it
are dropped from the index. Naps and the long wake stretches of 18–20 h
recordings fall outside.

```{eval-rst}
.. autoclass:: physioex.models.msvq.MSVQ
   :no-index:
   :members: quantize, forward, encode, draw_spectrum, draw_patch
   :show-inheritance:

.. autofunction:: physioex.models.msvq.compute_msvq_targets
   :no-index:
```
