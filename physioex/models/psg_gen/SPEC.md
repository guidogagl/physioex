# psg_gen: structured generative model of the PSG signal (specification)

Branch `feat/psg-gen`. Source of the design: `sleep-lens/docs/PROGETTI_GENERATIVI.md` (three designs P1, P2, P3 that
converge on one common core) under the requirements R1-R17 of `sleep-lens/docs/CRITICA_GENERATIVA.md`. The designers'
prototypes are in `_proto/` (`p1lib.py`, `prior_constants.py`, `p2lib.py`, `p3lib.py`, `p3infer.py`): they are the
reference implementations to port, not code to import. `_proto/` is removed before the merge.

## 0. Conventions

- `fs = 100`, epoch = 30 s = 3000 samples, token = 0.5 s (`n_tok = 60` per epoch). Signals are the cached clean
  channels in RobustScale units (median/MAD per night, clip 20): the AASM 75 uV threshold is `3.0` units
  (`msae_factor.rule_masks(swa_p2p=3.0)`).
- Everything is batched torch on the model device. Shapes: an epoch batch is `(B, T)` with `T = 3000`; a model
  input in the codec convention is `(B, L, C, T)` (L context epochs, C channels, centre epoch at `L // 2`).
- Classes: `CLS_BURST = 0`, `CLS_TRANSIENT = 1`, `CLS_SLOWWAVE = 2` (`CLASSES = ("burst", "transient", "slowwave")`).
- Background bands: `BG_BANDS = physioex.models.msvq.BANDS` (six bands, 0.5-30 Hz). Burst bands:
  `BURST_BANDS = {"theta": (4, 8), "alpha": (8, 11), "sigma": (11, 16), "beta": (16, 30)}` (alpha/sigma disjoint).
- No stage labels anywhere in training or inference. No sweeps: every hyper-parameter has a prior (section 8).
- Style: the surrounding code (`msae_factor.py`, `msvq.py`): type hints, short English docstrings, no prints,
  torch only (numpy allowed at I/O boundaries), no new dependencies.

## 1. Generative model (common to P1, P2, P3)

For one channel of one night, `x(t) = g * [ b(t) + sum_j a_j phi_{k_j}(t - tau_j) + sum_i c_i psi_{m_i}(t - sigma_i)
+ sum_l s_l chi_{q_l}(t - rho_l) ] + eps(t)`.

- `g`: scalar per night and channel, 20th percentile of the log-RMS of 3-4 s windows over a 20-min rolling span
  (R7). Not learned.
- `b(t)`: stationary background per window (4 s, hop 1 s): `logP[w, band]` = 20th percentile of the per-segment
  (1 s, hop 0.5 s) log band power of `x - events` over the window and a 6 s context, with the event supports
  masked out and a closed-form Gamma/chi-square correction for the quantile bias (R1). Not learned. The synthesised
  background for a decoded signal is band-shaped noise with those powers (a "floor"); `eps` is measured
  against `x - events` and reported per window as floor + excess.
- bursts `phi`: analytic Gabor atoms per band (cos/sin pair, >= 3 cycles, spectral concentration >= 0.8 in the
  band), parametric grid `(band, f on a 1 Hz grid, duration in cycles (3, 5, 8, 12), d <= 3 s; alpha also 5 and 8 s)`.
- transients `psi`: zero-mean, <= 1 s, >= 80 % of energy below 5 Hz (fixed bank of K-complex-like shapes at
  scales 0.5, 0.7, 0.9 s with negative-then-positive lobes; P2 learns 16 free shapes under the same constraints).
- slow waves `chi`: one cycle of 1-2.5 s (periods 1.0, 1.25, 1.5, 2.0, 2.5 s), zero-mean, low-pass 2 Hz; an event
  counts for the AASM 20 % rule only if its peak-to-peak amplitude in signal units is >= 3.0.
- Residual `eps`: measured, never modelled.

Inference is sequential and shared (R1, R5): (1) slow waves on the whitened raw epoch, (2) transients on the residual,
(3) bursts on the residual, each step = normalised-correlation scores of the whitened residual with the class
dictionary, selection above a per-atom null threshold with non-maximum suppression (within +- d/2 of the same
class), a per-epoch budget per class, joint least squares of the amplitudes on the selected support (explaining
away, ridge 1e-3, amplitude cap 5 x window RMS), then the background from the masked residual, then a second pass
of (1)-(3) re-whitened with the new background. Onsets at sample resolution (analytic atoms give amplitude and
phase from the cos/sin pair). Periodic firing of one transient atom (>= 4 events, CV of inter-onset intervals
< 0.12) is flagged `periodic` and excluded from counts (ECG).

## 2. Modules and interfaces

### `dicts.py`

```python
FS = 100
CLASSES = ("burst", "transient", "slowwave"); CLS_BURST, CLS_TRANSIENT, CLS_SLOWWAVE = 0, 1, 2
BG_BANDS  # = msvq.BANDS
BURST_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 11.0), "sigma": (11.0, 16.0), "beta": (16.0, 30.0)}

@dataclass
class Dictionary:
    atoms: torch.Tensor      # (K, L) float32, unit L2 norm, zero-padded to the common length L, centred (peak or envelope centre at L // 2)
    quad: torch.Tensor       # (K, L) the quadrature (sin) partner for bursts, zeros for the other classes
    cls: torch.Tensor        # (K,) long in {0, 1, 2}
    band: torch.Tensor       # (K,) long, index into BURST_BANDS order for bursts, -1 otherwise
    freq: torch.Tensor       # (K,) float, centre frequency in Hz for bursts, nan otherwise
    dur: torch.Tensor        # (K,) float, duration in seconds (90 % energy support)
    family: torch.Tensor     # (K,) long, family id: bursts share one family per (band, freq) across durations; other atoms one family each
    names: list[str]         # K human-readable names, e.g. "sigma_13Hz_5cyc", "kc_0.7s", "sw_1.5s"
    def to(self, device) -> "Dictionary"; def __len__(self) -> int
    @property n_family -> int

def gabor_bank(bands=BURST_BANDS, f_step=1.0, cycles=(3, 5, 8, 12), d_max=3.0, alpha_long=(5.0, 8.0), fs=FS) -> Dictionary
def transient_bank(scales=(0.5, 0.7, 0.9), fs=FS) -> Dictionary        # zero-mean, <= 1 s, >= 0.8 energy < 5 Hz
def slowwave_bank(periods=(1.0, 1.25, 1.5, 2.0, 2.5), fs=FS) -> Dictionary
def concat(*ds: Dictionary) -> Dictionary                              # re-pads to the max length, re-indexes families
def full_dictionary(fs=FS, **kw) -> Dictionary                         # slowwave + transient + gabor
def random_dictionary(like: Dictionary, seed: int) -> Dictionary       # same classes/durations/bands, random band-limited waveforms (null column, R13)
def spectral_concentration(atom: Tensor, band: tuple, fs=FS) -> float
def energy_below(atom: Tensor, hz: float, fs=FS) -> float
def coherence(d: Dictionary, same_class_only=True) -> Tensor           # (K, K) max |normalised cross-correlation| over lags
def overlap_report(d: Dictionary, fs=FS) -> dict                       # {"burst_energy_below_5hz": max, "transient_energy_below_5hz": min, "coh_burst_transient": max, "coh_burst_slowwave": max, "coh_transient_slowwave": max, "min_cycles": min over bursts}
def project_constraints(d: Dictionary) -> Dictionary                   # re-impose zero-mean, low-pass, duration and unit norm on learned atoms (used by P1/P2 learning)
```

### `background.py`

```python
def seg_band_logpow(x: Tensor, fs=FS, bands=BG_BANDS, seg=1.0, hop=0.5) -> Tensor      # (B, T) -> (B, n_seg, n_bands) log10 band power per segment (Hann, rfft)
def q20_gamma_factor(n_bins: int, q=0.2) -> float                                       # closed-form log10 correction of the q-quantile of a chi-square-like periodogram average (see _proto/p3lib.q20_gamma_factor and _proto/p1/prior_constants.py)
def background_estimate(x: Tensor, mask: Tensor | None, fs=FS, bands=BG_BANDS, win=4.0, hop=1.0, ctx=6.0, q=0.2, correct=True) -> dict
    # x (B, T), mask (B, T) bool True on event supports (ignored segments) or None
    # -> {"logP": (B, n_win, n_bands), "centers": (n_win,) seconds, "n_valid": (B, n_win, n_bands)}; windows with < 2 valid segments fall back to the epoch-level quantile
def whitening_filters(x: Tensor, L: int, fs=FS, q=0.2) -> tuple[Tensor, Tensor]       # (B, T) -> (W (B, L) zero-phase FIR from the robust (q-quantile) PSD, psd (B, F)); port of _proto/p1lib.whitening_filters
def whiten(x: Tensor, w: Tensor) -> Tensor                                               # FFT convolution, same length
def gain_g(x_night: Tensor, fs=FS, win=3.0, span_min=20.0, q=0.2) -> Tensor              # (N,) samples of a night -> (n_span,) one scalar log-RMS gain per 20-min span (last span padded); night_gain = median
def change_indicator(logP: Tensor, z=3.0) -> Tensor                                      # (B, n_win, n_bands) -> (B, n_win) float in [0, 1]: standardised jump of the band vector between consecutive windows, thresholded at z sigma (R10)
def synth_background(logP: Tensor, fs=FS, T=3000, win=4.0, hop=1.0, bands=BG_BANDS, seed: int | None = None) -> Tensor   # band-shaped noise with the given powers, overlap-add with Hann weights -> (B, T)
def floor_excess(resid: Tensor, logP: Tensor, fs=FS, bands=BG_BANDS, win=4.0, hop=1.0) -> dict   # resid = x - events: {"eps": (B, n_win) excess energy above the floor / window energy, "floor": (B, n_win), "excess": (B, n_win, n_bands) log10 ratio}
```

### `inference.py`

```python
@dataclass
class Events:                     # flat, sorted by (epoch, onset)
    epoch: Tensor    # (n,) long, index in the batch
    atom: Tensor     # (n,) long, index into the Dictionary
    cls: Tensor      # (n,) long
    onset: Tensor    # (n,) float seconds from the epoch start, sample resolution, centre of the atom
    amp: Tensor      # (n,) float, signed amplitude in signal units (after joint LS)
    phase: Tensor    # (n,) float radians (bursts), 0 otherwise
    dur: Tensor      # (n,) float seconds
    score: Tensor    # (n,) float normalised correlation at selection
    periodic: Tensor # (n,) bool
    channel: Tensor  # (n,) long (0 for single channel)
    def to(device); def __len__; def select(mask: Tensor) -> "Events"; def by_class(cls: int) -> "Events"
    @staticmethod def cat(*evs) -> "Events"; @staticmethod def empty(device) -> "Events"
    def to_dict(self) -> dict[str, list]   # JSON-serialisable (per-night dump)

def null_thresholds(d: Dictionary, bg: Tensor, fs=FS, q=0.95, n_max=192) -> Tensor      # (K,) per-atom 95th percentile of the normalised correlation on pure-background epochs bg (B, T), whitened
def sparse_events(x: Tensor, d: Dictionary, thresholds: Tensor, budget: dict[int, int], fs=FS, whiten_w: Tensor | None = None,
                  order=(CLS_SLOWWAVE, CLS_TRANSIENT, CLS_BURST), amp_cap=5.0, ridge=1e-3, nms=True, explain=True,
                  sw_p2p_min: float | None = None) -> tuple[Events, Tensor]
    # one pass of the sequential selection on x (B, T); returns (events, residual (B, T)); budget per class per epoch
    # (keys CLS_*); amplitude cap in multiples of the window RMS; explain=False -> matching-pursuit amplitudes (ablation)
def reconstruct(ev: Events, d: Dictionary, B: int, T: int, fs=FS, classes: Sequence[int] | None = None, exclude_periodic=False) -> Tensor   # (B, T)
def support_mask(ev: Events, d: Dictionary, B: int, T: int, fs=FS, pad=0.0) -> Tensor       # (B, T) bool
def periodic_flag(ev: Events, cv_max=0.12, min_count=4) -> Tensor                           # (n,) bool, per (epoch, atom)
def counts(ev: Events, B: int, exclude_periodic=True) -> Tensor                             # (B, 3) events per class
def decompose(x: Tensor, d: Dictionary, thresholds: Tensor, budget: dict[int, int], fs=FS, passes=2, bands=BG_BANDS, **kw) -> dict
    # the full pipeline on an epoch batch: whitening from the robust PSD -> sparse_events -> background_estimate on the masked
    # residual -> re-whiten with the background PSD -> sparse_events again (passes times). Returns
    # {"events": Events, "logP": (B, n_win, n_bands), "centers", "wave_events": (B, T), "resid": (B, T), "mask": (B, T),
    #  "eps": (B, n_win), "change": (B, n_win), "counts": (B, 3)}
def topk_per_token(x: Tensor, d: Dictionary, k=2, token=2.0, fs=FS) -> Events                # the proposal's baseline (ablation, R13)
```

### `model.py`

```python
class PSGGenBase(nn.Module):
    """Common encode/decode/intervene surface. Sub-classes override `decompose_batch` and `loss`."""
    def __init__(self, dictionary: Dictionary, budget: dict[int, int], thresholds: Tensor, fs=FS, sequence_length=41, token_sec=0.5, n_sites=1)
    # buffers: atoms, quad, thresholds; attributes: d (Dictionary on device), n_tok=60, d_p, d_e, d_n, vq_levels={}, n_codes={}, site_of_dataset=None (as msae_factor)
    def decompose_batch(self, x: Tensor) -> dict                    # (B, T) -> inference.decompose(...) output (sub-classes may add keys)
    def tokens(self, dec: dict, B: int) -> Tensor                   # (B, n_tok, d_p): per 0.5 s token [logP of the covering window (6); summed |amp| per family over events whose support covers the token (n_family); counts per class in the token (3)]
    def epoch_state(self, dec: dict, B: int) -> Tensor              # (B, d_e): [mean logP over windows (6); counts per class (3); mean change (1); mean eps (1)]
    def night_state(self, x_ctx: Tensor) -> Tensor                  # (B, L, T) -> (B, d_n): [log g of the context (1); 20/50/80 quantiles of the per-window log-RMS (3)] (d_n = 4)
    def encode(self, x: Tensor) -> dict                             # (B, L, C, T) -> {"N": (B, d_n), "E": (B, d_e), "P": (B, n_tok, d_p), "codes": {}}, channel 0 only; the names follow msae_factor so pretrain_combo.encode_night and concept_scan work unchanged
    def decode(self, dec: dict, parts=False, classes=None, seed=None, exclude_periodic=False) -> Tensor | tuple
        # events part = inference.reconstruct(...), background part = background.synth_background(logP, seed); returns the sum, or (events, background) with parts=True
    def intervene(self, dec: dict, kind: str, **kw) -> dict
        # kind in {"remove_class" (cls), "remove_family" (family), "swap_background" (logP_from: another dec), "scale" (factor), "keep_only" (cls)}; returns a new dec (events filtered / logP replaced), never mutates the input
    def forward(self, x: Tensor) -> dict                            # {"N", "E", "P", "wave", "wave_det", "wave_bg", "dec"}; wave = decode(dec)
    def loss(self, out: dict, xc: Tensor, site: Tensor | None) -> tuple[Tensor, Tensor, dict]   # (loss, acc, extra) as MSAEFactorTrainer._losses; base = zero loss, acc = R^2 of band log power of the decoded epoch
    def bits_per_epoch(self) -> int

def fidelity_gate(x: Tensor, x_hat: Tensor, fs=FS, n_tok=60) -> dict   # gate 0 (R12): F1 per rule of msae_factor.rule_masks between x and x_hat with the thresholds frozen on x (the KC median and the spindle mu/sd are computed on x and reused on x_hat), plus the residual energy share
```

`rule_masks` with frozen thresholds: add to `msae_factor.rule_masks` an optional argument `ref: Tensor | None` (the
reference signal whose statistics define the thresholds); default behaviour unchanged.

### `p1.py` (classical, P1)

`PSGGenP1(PSGGenBase)`: no learnable parameters. `fit_transients(loader_or_batches, n_iter=2, merge_coh=0.9)` learns
the transient atoms by alternating minimisation (E-step: `sparse_events` with the current dictionary; M-step: least
squares of each transient atom on its selected segments, projection with `project_constraints`, restart of unused atoms
from the highest-residual segments, merge of atoms with coherence > 0.9), then recomputes the thresholds. Ablation
flags in the constructor: `whiten=True`, `use_gain=True`, `fixed_dict=True`.

### `p2.py` (structured neural, P2)

`PSGGenP2(PSGGenBase)`: learnable per-atom thresholds and step sizes of a block-ISTA unrolled 10 iterations with a
dictionary-tied analysis bank (candidates), free transient (16) and slow-wave (16) shapes re-projected on the
constraints after every optimiser step, a night-adversary head (gradient reversal, `msae_factor.reverse`) on the
per-epoch presence vector (binary per family), prices per class (L1 per unit duration for bursts, L1 for transients and
slow waves) calibrated so equal energy costs the same, Gram coherence penalty (< 0.3) and periodicity penalty. Selection,
NMS, budgets and joint LS as the common `inference` (the unrolled codes only propose candidates). Loss (two residuals,
R4): time-domain MSE of `x - events` restricted to +-1 s around the supports, plus masked log band power of the residual
against `logP` with a chi-square dead zone, plus the prices, adversary (weight 0.2), Gram and periodicity; weights set
once by equal gradient norm at step 0 and logged. Ablation flags: `fixed_dict`, `no_adv`, `events_only` (no background
term). Port of `_proto/p2lib.CSC`.

### `p3.py` (probabilistic, P3)

`PSGGenP3(PSGGenBase)`: background as a per-band state (6 log levels) with a random walk plus sparse jumps (Kalman
smoother with a jump indicator, `change` = posterior jump probability), Gamma likelihood of the masked periodogram given
the state, events as a marked point process with a per-class intensity (prior mean rate per epoch from the budgets), a
refractory term (NMS) and an anti-periodicity term; selection by likelihood gain with a closed-form Beta/Bonferroni
false-alarm gate (alpha 0.05) instead of fixed thresholds; amplitudes by GLS under the background PSD; two passes.
Loss: NLL of the two residuals (whitened time domain on supports + Gamma on the masked bands) + KL of the presence
process to the prior + night adversary on presences. The amortised proposal CNN (`ProposalCNN`, 1-D conv over the
whitened epoch, cells of 0.1 s, Concrete relaxation) is an optional module trained from the explicit codifier's
selections (supervised start) and used only as the `events_only`/`amortised` ablation; the explicit inference is
the model. Port of `_proto/p3lib.py` and `_proto/p3infer.py`.

### `trainer.py`

`PSGGenTrainer(Trainer)` with the `_step` of `MSAEFactorTrainer` (stack_channels, centre epoch, `sites_of`, autocast)
calling `model.loss(out, xc, site)`. For P1 `_step` returns the base loss (no optimiser step needed; the trainer still
logs `acc`). Checkpoint = `state_dict` (buffers included).

### `physioex/train/bin/psg_gen.py` (CLI)

`python -m physioex.train.bin.psg_gen fit --model p1|p2|p3 --variant full|fixed|random|noadv|events_only --cache ...
--datasets mass_ss01,mass_ss03,mass_ss04,mass_ss05 --out <dir> --steps 10000 --batch 64` and `... dump --ckpt ... --nights ...`
writing per night `events_<mod>.json` (Events.to_dict), `bg_<mod>.npz` (logP, centers, eps, change, g) and the
`N/E/P` arrays in the format of `pretrain_combo.encode_night`, plus `gate0.json` from `fidelity_gate` on the dump
nights. Reuse the dataset/loader construction of `physioex/train/bin/msae_factor_fit.py` if it exists, else of
`pretrain_combo.py`. `run.json` with git sha, config hash, host, job id as the other tools.

## 3. Tests (`tests/models/test_psg_gen.py`)

Synthetic data only (`dicts`-based generators: 1/f background, planted spindles 11/13/15 Hz, K-complexes, slow-wave
trains, alpha runs, ECG trains), all on CPU and < 2 min total, markers as the other model tests:

1. dictionary constraints: every burst >= 3 cycles and concentration >= 0.8; transients zero-mean, <= 1 s, >= 0.8 energy
   below 5 Hz; slow waves 1-2.5 s; `overlap_report`: `coh_burst_transient < 0.3`, `coh_burst_slowwave < 0.3`.
2. `q20_gamma_factor`: on white noise the corrected q20 band power is within 0.3 dB of the true power.
3. background bias: dense synthetic N3 (12 planted slow waves per epoch), `background_estimate` with the event mask vs
   without: masked bias in 0.5-2 Hz < 1.5 dB, unmasked > 3 dB.
4. planted spindles: `decompose` recovers >= 80 % of planted spindles at 11/13/15 Hz within +-0.3 s with the right band.
5. explaining away: two spindles 0.4 s apart and a spindle on the tail of a K-complex: <= 1.2 events per planted event
   with `explain=True, nms=True`, > 1.5 with `topk_per_token`.
6. absorption: on pure 1/f epochs the events' energy share is < 0.05.
7. ECG: a 60-bpm transient train is flagged periodic and excluded from `counts`.
8. `encode` shapes: `P (B, 60, d_p)`, `E (B, d_e)`, `N (B, 4)`; `decode(parts=True)` parts sum to `decode()`;
   `intervene("remove_class", cls=CLS_BURST)` zeroes the sigma band power of the events part; `fidelity_gate` on
   `x` vs `decode(dec)` gives F1 >= 0.9 for `swa` and `spindle` on the synthetic set.
9. P1 `fit_transients` on synthetic K-complexes: correlation of the learned atom with the true shape >= 0.95.
10. P2 and P3: one training step on 8 synthetic epochs runs, the loss is finite, the extra dict has the logged term
    ratios; 200 steps on 8 epochs: time-domain R^2 on the supports > 0.8 (P2) / NLL decreases (P3).

## 4. Data, split and cost (R16)

Training pool MASS SS01, SS03, SS04, SS05 (181 nights), SS02 frozen for evaluation; split by night; <= 3 passes
(~10k steps at batch 64); one sbatch with four runs in parallel on one H200 (`full`, `fixed`/`random`, `noadv`,
`events_only`), <= 3 h per project.

## 5. Hyper-parameters fixed by prior (R13)

Budgets per epoch per class from the 99th percentile of the rule-detector counts on MASS SS01 (P2 test j): transients 4
(KC 2 + margin), slow waves 20, bursts 14 (alpha 14, sigma 4, theta 3, beta 2 by band when grouped). Null thresholds
= 95th percentile of the normalised correlation on pink-noise epochs (P1 test k: alpha 0.45, beta 0.63, sigma 0.60,
slow wave 0.50, theta 0.55, transient 0.57). Background window 4 s, hop 1 s, context 6 s, quantile 0.2 with the Gamma
correction. Gain: q20 of the 3 s log-RMS over 20 min. Amplitude cap 5 x window RMS. NMS +- d/2. Periodic: CV < 0.12,
>= 4 events. P2: ISTA 10 iterations; P3: false-alarm alpha 0.05.
