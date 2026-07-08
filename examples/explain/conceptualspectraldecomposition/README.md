# CSD (Conceptual Spectral Decomposition) Example

Complete example of using CSD to explain foundation model predictions for sleep staging.

## Overview

**CSD (Conceptual Spectral Decomposition)** explains foundation model embeddings by:

1. Identifying class-specific embedding dimensions using a specificity strategy
2. Running SpectralGradients independently on each selected dimension
3. Producing per-concept time-frequency attribution maps
4. Aggregating into a class-level explanation

This example uses:
- **Dataset**: MASS SS03 (~62 subjects, 30-second epochs, AASM scoring)
- **Model**: CBRAMod (D=200 embeddings, pure EEG encoder)
- **Linear Probe**: Trained on random subject split (LayerNorm + Linear)

## Files

- `utils.py` - LinearProbeWithLN module and training/loading utilities
- `train_probe.py` - Train linear probe on MASS SS03
- `explain_csd.py` - Run CSD explanation and generate visualizations
- `visualize.py` - Matplotlib visualization functions

## Usage

### 1. Train Linear Probe

First, train a linear probe on MASS SS3:

```bash
python examples/explain/conceptualspectraldecomposition/train_probe.py \
    --gpu_id 0 \
    --output_dir ./csd_checkpoints \
    --train_ratio 0.7 \
    --max_epochs 50
```

This will:
- Load MASS SS03 dataset
- Create random 70/30 train/valid split of subjects
- Extract CBRAMod embeddings
- Train LinearProbeWithLN (LayerNorm + Linear)
- Save probe weights to `./csd_checkpoints/probe.pt`
- Save metrics to `./csd_checkpoints/metrics.json`

Expected output:
```
==============================================================
CSD Linear Probe Training: cbramod on mass_ss03
==============================================================

[1] Loading dataset...
  62 subjects found
  Channels: ['EEG C3-CLE', 'EOG Left Horiz', 'EMG Chin1']

[2] Splitting subjects (train_ratio=0.7)...
  Train: 43 subjects
  Valid: 19 subjects

[3] Loading cbramod encoder...
  Parameters: 1,234,567
  Embedding dim: 200

[4] Training linear probe...
  ...
  Final: ACC=0.8234, MF1=0.7812, kappa=0.7698
```

### 2. Run CSD Explanation

Then, explain predictions for a specific subject:

```bash
python examples/explain/conceptualspectraldecomposition/explain_csd.py \
    --gpu_id 0 \
    --probe_path ./csd_checkpoints/probe.pt \
    --subject_id 01-01-0001 \
    --target_class 3 \
    --output_dir ./csd_results
```

Arguments:
- `--probe_path`: Path to trained probe.pt
- `--subject_id`: Subject ID to explain (default: first subject)
- `--target_class`: Sleep stage to explain (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
- `--output_dir`: Directory for results
- `--max_concepts`: Limit number of concepts (default: all above threshold)
- `--specificity_tau`: Specificity strategy tau parameter (default: 0.5)
- `--freq_step`: Frequency band width in Hz (default: 4.0)
- `--mask_threshold`: Concept selection threshold (default: 0.5)
- `--top_k_plots`: Number of top concepts to visualize (default: 10)

### 3. Results

The script generates:

**Data files:**
- `metadata.json` - Explanation metadata and concept info
- `class_attribution.npy` - Class-level attribution map (n_bands, T)

**Visualizations:**
- `class_attribution.png` - Aggregated class heatmap
- `top_concepts.png` - Bar chart of top concepts by weight
- `per_channel_energy.png` - Per-channel attribution energy
- `csd_summary_N3.png` - Summary figure with all plots
- `top_concept_*.png` - Individual concept heatmaps

## Example Output

```
==============================================================
CSD Explanation: cbramod on mass_ss03
==============================================================

[1] Loading dataset...
  Subject: 01-01-0001
  Target class: 3 (N3)

[2] Loading cbramod encoder...
  Embedding dim: 200

[3] Loading probe from ./csd_checkpoints/probe.pt...
  Probe loaded: W shape = torch.Size([5, 200])

[4] Loading subject signal...
  Signal shape: torch.Size([987, 3, 3000]) (N=987 scored epochs)
  Channels: ['EEG C3-CLE', 'EOG Left Horiz', 'EMG Chin1']

[5] Creating CSD explainer...
  Freq step: 4.0 Hz
  Specificity: Margin (tau=0.5)
  Mask threshold: 0.5

[6] Running CSD explanation...
[CSD] class=3, 12 concepts selected (threshold=0.5, strategy=margin)

  Top 5 concepts:
    Dim 47: W=+0.234, mask=0.87, top_freq=1.5 Hz
    Dim 123: W=+0.189, mask=0.82, top_freq=2.5 Hz
    Dim 89: W=-0.145, mask=0.79, top_freq=14.5 Hz
    ...

[7] Saving results to ./csd_results...
  Saved: metadata.json
  Saved: class_attribution.npy

[8] Generating visualizations...
  Saved: class_attribution.png
  Saved: top_concepts.png
  Saved: per_channel_energy.png
  Saved summary: csd_summary_N3.png
  Saved: 12 individual concept heatmaps
```

## Interpretation

**Per-concept maps**: Each concept (embedding dimension) shows which time-frequency
patterns in the input signal drive that dimension. The top band frequency indicates
the most important frequency range.

**Class attribution**: Aggregated map shows the weighted sum of all concepts,
answering "why was class N3 predicted?" in time-frequency.

**Top concepts**: Bar chart shows which dimensions contribute most to the class
prediction, with their probe weights and specificity mask values.

## Specificity Strategies

CSD supports multiple specificity strategies (see `physioex.explain.foundational.specificity`):

- `MarginSpecificity` (default): Dynamic, per-input class margin
- `CohenDSpecificity`: Static, effect size across dataset
- `SoftmaxSpecificity`: Dynamic, softmax-normalized contributions
- `TopKSpecificity`: Hard selection of top-K dimensions
- `NoFilter`: Baseline, all dimensions

Change via `--specificity_tau` or modify the script to use a different strategy.

## Requirements

- `PHYSIOEX_DATA` environment variable pointing to MASS/Original directory
- MASS SS03 cohort (SS03) data available
- GPU recommended (but CPU works)
