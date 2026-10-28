# `ProtoSleepNet` and `ProtoSleepTransformer`

Two pretrained sleep stagers, trained on 28 sites from 12 public and clinical datasets and published on the Hugging Face Hub
in [`4rooms/physioex`](https://huggingface.co/4rooms/physioex):

| Hub name | Epoch encoder | Sequence length `L` | Parameters |
|---|---|---:|---:|
| `protosleepnet-gagliardi` (PSN) | SeqSleepNet | 20 | 0.79 M |
| `protosleeptransformer-gagliardi` (PST) | SleepTransformer | 21 | 3.98 M |

Both are `physioex.models.protosleepnet.ProtoSleepNet` networks: a per-channel epoch encoder, a channel mixer
that learns how much to trust each modality, a residual sequence encoder and a linear classifier. They are the models of
the paper *Prototype-based interpretable sleep staging with physiologically meaningful sub-stage pattern discovery*
(npj Digital Medicine; code and analyses in [guidogagl/protosleepnet](https://github.com/guidogagl/protosleepnet)),
retrained on many sites at once. The weights of the paper itself (one dataset per model) are a separate archive.

## Quick start

```python
import torch
from physioex.models import load_from_pretrained

model = load_from_pretrained("protosleeptransformer-gagliardi", verbose=True)   # eval mode, CPU

# (batch, L, channels, T, F): log-power STFT spectrograms of EEG, EOG, EMG (T=29, F=129)
x = torch.randn(2, 21, 3, 29, 129)
with torch.no_grad():
    logits = model(x)            # (2, 21, 5): W, N1, N2, N3, REM
```

`verbose=True` prints the per-site test metrics stored with the weights. The input is what the `seqsleepnet`
[pipeline preset](../../preprocess.md) produces for the channels `["EEG", "EOG", "EMG"]`, so the models can be fed
directly from any physioex dataset:

```python
from physioex.data import get_dataset

ds = get_dataset("hmc")(channels=["EEG", "EOG", "EMG"], pipelines="seqsleepnet", sequence_length=21)
```

`encode(x)` returns the contextualised epoch embeddings `(B, L, d_model)`. `forward(x, quantize=True)` replaces the epoch
embeddings by their nearest prototype, but it needs a codebook (`set_codebook`) that is learned after training from the
epoch embeddings; the checkpoints on the Hub contain no codebook, and `quantize=False` (the default) is the plain
classifier. The prototype analyses of the paper are in its repository, and relevance maps for these models are in
[Explainability](../../explain/explain.md) (LRP supports the default, non-quantised forward).

## Training data

Both models were trained jointly on the training subjects of 28 sites. The test and validation subjects of every site are
disjoint from the training subjects (split by subject, per site).

| Dataset | Sites | Source |
|---|---|---|
| Sleep-EDF | `sleepedf` | PhysioNet |
| HMC | `hmc` | PhysioNet |
| DCSM | `dcsm` | ERDA / University of Copenhagen |
| MASS | `mass_ss01` ... `mass_ss05` (cohorts SS1-SS5) | CEAMS |
| SHHS | `shhs` | NSRR |
| MESA | `mesa` | NSRR |
| MrOS | `mros` | NSRR |
| HomePAP | `hpap` | NSRR |
| WSC | `wsc` | NSRR |
| STAGES | `stages_BOGN`, `stages_GSBB`, `stages_GSDV`, `stages_GSLH`, `stages_GSSA`, `stages_GSSW`, `stages_MSMI`, `stages_MSNF`, `stages_MSQW`, `stages_MSTH`, `stages_MSTR`, `stages_STLK`, `stages_STNF` (13 clinical sites) | NSRR |
| Alzheimer's disease study, healthy controls | `alzheimers_hc` | UZ Leuven |
| Parkinson's disease study, healthy older adults, night recordings | `parkinsons_night_hoa` | UZ Leuven |

Recipe (the same for both models): 3 channels (EEG, EOG, EMG), `seqsleepnet` pipeline, site-balanced sampling
(`make_site_sampler`, every site gets the same probability mass, see [Data](../../data.md)), batch size 32, learning rate
1e-4 with a reduce-on-plateau scheduler, no weight decay, 300,000 windows per epoch, at most 20 epochs, channel dropout 0.5
in the channel mixer, seed 42. The released checkpoint is the one with the highest macro-averaged validation kappa among the
evaluated checkpoints (up to 30 validation subjects per site): step 161,500 (epoch 17) for PSN and step 129,000 (epoch 13)
for PST. Neighbouring checkpoints differ by about 0.001 kappa, so the choice is not critical.

## Performance by training site

Metrics are computed on the **test subjects of each site**, pooling the epochs of all of them (this is not a mean over
nights), with full-night sliding-window voting (windows of `L` epochs at `L` different offsets, overlapping predictions averaged). Accuracy, Cohen's kappa and macro F1 over the five AASM
stages. Read the tables with these limits in mind:

- The sites were seen in training (other subjects of the same sites), so this is **not** external validation: a new site,
  montage or population can perform differently.
- Single training run (one seed), no confidence intervals. Several sites have 2-6 test subjects, so their values are noisy;
  SHHS is by far the largest test set (842 nights).
- Per-site values mix population, scoring protocol and recording hardware; do not read a difference between two sites as a
  difference in model quality without checking the labels.

The two single-centre studies at the end of the training table (`alzheimers_hc`, `parkinsons_night_hoa`) were part of training but their per-site
metrics are not published, so the tables below list 26 of the 28 sites.

### ProtoSleepNet (`protosleepnet-gagliardi`)

| Site | Test subjects | Accuracy | Cohen's kappa | Macro F1 |
|---|---:|---:|---:|---:|
| `dcsm` | 26 | 0.867 | 0.809 | 0.774 |
| `hmc` | 15 | 0.814 | 0.754 | 0.784 |
| `hpap` | 25 | 0.847 | 0.790 | 0.801 |
| `mass_ss01` | 5 | 0.811 | 0.724 | 0.761 |
| `mass_ss02` | 2 | 0.910 | 0.858 | 0.837 |
| `mass_ss03` | 6 | 0.888 | 0.835 | 0.844 |
| `mass_ss04` | 4 | 0.871 | 0.814 | 0.834 |
| `mass_ss05` | 3 | 0.891 | 0.841 | 0.821 |
| `mesa` | 206 | 0.842 | 0.776 | 0.771 |
| `mros` | 219 | 0.859 | 0.795 | 0.757 |
| `shhs` | 842 | 0.872 | 0.818 | 0.780 |
| `sleepedf` | 15 | 0.842 | 0.778 | 0.794 |
| `stages_BOGN` | 8 | 0.865 | 0.800 | 0.806 |
| `stages_GSBB` | 3 | 0.814 | 0.729 | 0.691 |
| `stages_GSDV` | 19 | 0.804 | 0.679 | 0.697 |
| `stages_GSLH` | 3 | 0.812 | 0.732 | 0.740 |
| `stages_GSSA` | 5 | 0.826 | 0.696 | 0.673 |
| `stages_GSSW` | 8 | 0.836 | 0.722 | 0.711 |
| `stages_MSMI` | 6 | 0.840 | 0.764 | 0.787 |
| `stages_MSNF` | 4 | 0.854 | 0.735 | 0.742 |
| `stages_MSQW` | 15 | 0.805 | 0.711 | 0.734 |
| `stages_MSTH` | 3 | 0.823 | 0.745 | 0.771 |
| `stages_MSTR` | 29 | 0.849 | 0.779 | 0.800 |
| `stages_STLK` | 16 | 0.802 | 0.708 | 0.704 |
| `stages_STNF` | 49 | 0.691 | 0.591 | 0.630 |
| `wsc` | 235 | 0.833 | 0.735 | 0.729 |

### ProtoSleepTransformer (`protosleeptransformer-gagliardi`)

| Site | Test subjects | Accuracy | Cohen's kappa | Macro F1 |
|---|---:|---:|---:|---:|
| `dcsm` | 26 | 0.870 | 0.814 | 0.781 |
| `hmc` | 15 | 0.815 | 0.754 | 0.789 |
| `hpap` | 25 | 0.857 | 0.803 | 0.812 |
| `mass_ss01` | 5 | 0.810 | 0.723 | 0.766 |
| `mass_ss02` | 2 | 0.899 | 0.841 | 0.833 |
| `mass_ss03` | 6 | 0.894 | 0.843 | 0.846 |
| `mass_ss04` | 4 | 0.870 | 0.812 | 0.829 |
| `mass_ss05` | 3 | 0.888 | 0.838 | 0.824 |
| `mesa` | 206 | 0.842 | 0.778 | 0.778 |
| `mros` | 219 | 0.872 | 0.811 | 0.765 |
| `shhs` | 842 | 0.880 | 0.829 | 0.797 |
| `sleepedf` | 15 | 0.842 | 0.775 | 0.780 |
| `stages_BOGN` | 8 | 0.873 | 0.811 | 0.817 |
| `stages_GSBB` | 3 | 0.828 | 0.749 | 0.708 |
| `stages_GSDV` | 19 | 0.777 | 0.649 | 0.688 |
| `stages_GSLH` | 3 | 0.790 | 0.699 | 0.698 |
| `stages_GSSA` | 5 | 0.821 | 0.692 | 0.676 |
| `stages_GSSW` | 8 | 0.812 | 0.675 | 0.674 |
| `stages_MSMI` | 6 | 0.834 | 0.763 | 0.794 |
| `stages_MSNF` | 4 | 0.853 | 0.743 | 0.757 |
| `stages_MSQW` | 15 | 0.819 | 0.730 | 0.749 |
| `stages_MSTH` | 3 | 0.823 | 0.743 | 0.763 |
| `stages_MSTR` | 29 | 0.844 | 0.773 | 0.796 |
| `stages_STLK` | 16 | 0.812 | 0.724 | 0.717 |
| `stages_STNF` | 49 | 0.720 | 0.632 | 0.672 |
| `wsc` | 235 | 0.845 | 0.750 | 0.739 |

The same numbers are stored with the weights (`metrics.json` in each model folder on the Hub).

## Intended use and limits

For research on sleep staging and its interpretability; not a medical device, and not for clinical diagnosis. The models expect
the spectrogram input described above: a different preprocessing, signal amplitude scale or montage changes the input
distribution, so check the staging on a sample of your own data before relying on it. The channel set is fixed to EEG, EOG
and EMG (the channel mixer was trained with random channel dropout, which makes the models tolerant to a missing channel, but
this was not evaluated per site here).

## Reproducibility

Every model folder on the Hub has `config.json` with the factory call and the training summary, including the SHA-256 of
`model.pt`:

| Hub name | Checkpoint step | SHA-256 of `model.pt` |
|---|---:|---|
| `protosleepnet-gagliardi` | 161,500 | `1156a3fe18d3bc8a83a775bcc8e1a1ba08c0c4624b3415a4fc07982e907136db` |
| `protosleeptransformer-gagliardi` | 129,000 | `08d0fbc806fc08dc7d0167e1c2b81b0e2b4f5495bad2c1e5dda6bde14972e603` |

`physioex/examples/pretrained/protosleepnet/export_release.py` builds the folder from a training checkpoint and its per-site
evaluation; nothing is recomputed, and the metrics are copied unchanged.

## API

```{eval-rst}
.. autoclass:: physioex.models.protosleepnet.ProtoSleepNet
   :no-index:
   :members: from_sleep_transformer, from_seq_sleep_net, set_codebook, encode, forward
   :show-inheritance:
```
