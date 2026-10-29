# `MSAE`

`MSAE` is a continuous hierarchical autoencoder of single-channel clean EEG. For
the centre epoch `t` of a 41-epoch window it returns three latents:

- **N** (128 numbers): what the 40 neighbouring epochs predict about `t`, with `t` masked out. N is context: it is computed from the neighbours, so it costs nothing for epoch `t`;
- **E** (128 numbers): what N does not predict about `t`, such as the background and the spectrum. The E encoder sees N through FiLM;
- **P** (`T_p × d_p`, with 1 s or 0.5 s tokens): what N and E do not explain, such as spindles, K-complexes, transients and phase. The P encoder sees N and E.

The chain N → E → P is conditional coding, as in learned video codecs. Each
encoder sees the coarser latents and adds only what they miss. A single decoder
is called on the three prefixes, with missing latents set to zero, and each
prefix has its own loss:

| prefix | loss |
|---|---|
| `x_N = D(N)` | spectral envelope, STFT 2–4 s |
| `x_NE = D(N, E)` | envelope and spectrum, STFT 1–4 s |
| `x_NEP = D(N, E, P)` | full waveform: L1, STFT 0.25–4 s, phase |

Two heads also predict the epoch log-PSD from N and from N+E.

The model must compress. E and P go through `tanh` and an 8-bit channel: uniform
noise of one step in training and rounding in evaluation. With `d_p = 8` and
1 s tokens, epoch `t` costs 368 numbers (2,944 bits), against 3,000 samples
(48,000 bits at 16 bit). Encoder and decoder are SEANet stacks after EnCodec, with
a multi-scale stem after NeuroRVQ. The decoder draws the whole 30 s epoch at once.
Sleep stages are never used. `MSVQ` ({doc}`msvq`) is the discrete stage that will
sit on top of these latents.

`tokens(x)` gives the sequence a downstream reader such as a Q-Former consumes:
`(B, 2 + T_p, 128)` = `[N, E, P_1 … P_T]`.

Training uses the MSVQ script:

```bash
python -m physioex.train.bin.msvq --model msae --d-p 8 --token-sec 1.0 --dataset dcsm --out outputs/msae_dcsm/run0
```

At the end the script writes `recon_eval.json`, the reconstruction audit of the
test nights (`physioex.models.recon_eval`). It reports:

- R² per band, with phase and of the amplitude only;
- the gain of E and P per band;
- the cost of rounding the latents to 8, 6 and 4 bits;
- a PCA with as many components as the model's numbers, and a k-means on raw 1 s segments.

```{eval-rst}
.. autoclass:: physioex.models.msae.MSAE
   :no-index:
   :members: encode, tokens, decode, forward, bits_per_epoch
   :show-inheritance:

.. autoclass:: physioex.models.msae.MSAETrainer
   :no-index:

.. autofunction:: physioex.models.recon_eval.evaluate
   :no-index:
```
