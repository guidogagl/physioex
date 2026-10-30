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

## Losses and the high bands

Above about 5 Hz the phase cannot be predicted from a small budget. MSE, like a PCA,
answers with the mean, so those bands come out 3 to 50 times too weak. Three options
are available:

| option | N+E+P loss | effect |
|---|---|---|
| `--loss mse` (default) | MSE + 0.1 × multi-resolution STFT | best wave R², high bands too weak |
| `--loss band --low-hz f` | MSE of the wave low-passed at `f`; log band power per 1 s segment above `f` | exact power, phase above `f` invented |
| `--texture --pow-hz f` | MSE on the deterministic wave; band power on wave + texture | MSE wave plus a noise texture with the right power |

The band losses move their 1 s grid at random at every step. On a fixed grid the
model learnt the segment edges: R² 1.00 on the training grid and 0.67–0.86 on the
grid shifted by 0.5 s.

The texture is band-passed white noise times an amplitude per token and band. Its
gradient reaches the latents but never the deterministic path, so `wave_det` stays
the MSE estimate.

## Recommended configuration

On DCSM (30 epochs, 39 test nights), the recommended configuration is:

```bash
python -m physioex.train.bin.msvq --model msae --dataset dcsm --width 32 --token-sec 0.5 --d-p 16 \
    --texture --pow-hz 4.5 --band-weight 3 --grad-clip 1.0 --epochs 30 --out outputs/msae_dcsm/tex1088
```

It costs 1088 numbers per epoch (8,704 bits, 5.5× against 16-bit samples). Its scores:

- wave R² 0.963 (`wave_det`), against 0.945 for a PCA with the same numbers;
- phase R² 1.00 below 4 Hz;
- log band power over 5 s of 0.98 / 0.98 / 0.96 / 0.98 on 5–8 / 9–12 / 13–16 / 17–30 Hz.
  MSE alone gets 0.62 on 17–30 Hz, with half the power missing;
- staging κ 0.72 per night from a linear probe on [N; E] (log-PSD: 0.62);
- spindle proxy AUC 0.98 from the reconstruction.

At 608 numbers per epoch MSE alone still beats the PCA (0.907 against 0.879), but the
texture costs it its margin. At 368 numbers or fewer no variant clearly beats the PCA.

## Evaluation

At the end the script writes `recon_eval.json`, the reconstruction audit of the
test nights (`physioex.models.recon_eval`). It reports:

- R² per band: with phase, of the amplitude per bin, and of the log band power per second;
- the gain of E and P per band;
- the cost of rounding the latents to 8, 6 and 4 bits;
- a PCA with as many components as the model's numbers, and a k-means on raw 1 s segments.

Per 1 s, even a random-phase signal with the true power of each second reaches only
R² 0.28–0.71 on the log band power, because a 1 s segment holds only 4 to 14
frequency bins. The envelope metric is therefore `logpow5`, computed over 5 s.
`logpow_shift` repeats the 1 s metric on the grid shifted by 0.5 s.

`python -m physioex.models.latent_probe --run <dir>` writes `probe.json`. It holds
a staging probe (κ per test night) and a label-free spindle proxy. Sleep stages are
used only there, for evaluation.

```{eval-rst}
.. autoclass:: physioex.models.msae.MSAE
   :no-index:
   :members: encode, tokens, decode, forward, bits_per_epoch
   :show-inheritance:

.. autoclass:: physioex.models.msae.MSAETrainer
   :no-index:

.. autofunction:: physioex.models.recon_eval.evaluate
   :no-index:

.. autofunction:: physioex.models.latent_probe.evaluate
   :no-index:
```
