# PhysioEx Lectures

Hands-on teaching materials for the [PhysioEx](https://github.com/guidogagl/physioex) library.
Each lecture includes a self-contained Jupyter notebook with pretrained models, data loading, and explainability — ready to run.

## Available Lectures

### Explainable AI for Sleep Staging
**UGent — Explainable & Trustworthy AI** | June 24, 2026

A 20-minute walkthrough covering the full pipeline from EEG data to explained predictions:

1. **Data** — Load raw EEG (Sleep-EDF) and spectrograms (MASS) via a uniform API; visualize N3 deep sleep with power spectral analysis
2. **Models** — Run two pretrained sequence-to-sequence models (TinySleepNet, SeqSleepNet) and compare predictions
3. **Explainability** — Apply Integrated Gradients to both models; validate that spectrogram attributions concentrate on the delta band (0.5–4 Hz), matching known N3 physiology

| Material | Link |
|----------|------|
| Notebook | [`2026-06-24_ugent_xai/notebook.ipynb`](2026-06-24_ugent_xai/notebook.ipynb) |
| Slides (PDF) | [`2026-06-24_ugent_xai/slides.pdf`](2026-06-24_ugent_xai/slides.pdf) |

#### Quick start

```bash
pip install physioex
cd lectures/2026-06-24_ugent_xai
jupyter notebook notebook.ipynb
```

**Requirements:** Python 3.10+, PyTorch, a CUDA GPU (recommended for Integrated Gradients). Datasets and pretrained models are downloaded automatically on first use.

## Structure

```
lectures/
├── README.md
├── assets/              # Shared logos (PhysioEx, KU Leuven)
├── templates/           # Beamer slide template (.sty)
└── 2026-06-24_ugent_xai/
    ├── notebook.ipynb   # Main lecture notebook
    ├── slides.pdf       # Presentation slides
    ├── util.py          # Plotting utilities for the notebook
    ├── run_notebook.py  # Script to run all cells headless (for testing)
    └── outputs/         # Pre-generated plots (also embedded in notebook)
```

> **Note:** LaTeX source files (`.tex`, `.sty`) are not tracked in git. Only notebooks, slides PDFs, Python utilities, and assets are version-controlled.
