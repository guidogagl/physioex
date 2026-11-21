# How wav2sleep was trained

Authors: Carter, J. F. & Tarassenko, L. (2024). *wav2sleep: A Unified Multi-Modal
Approach to Sleep Stage Classification from Physiological Signals.*
arXiv:2411.04644 — code: https://github.com/joncarter1/wav2sleep

Physioex does not ship a training script for these models; the references below
point into the authors' repository.

## Data and split

- NSRR datasets CC-DB-HS (ccshs), CFS (cfs), CHAT (chat), MESA (mesa), SHHS
  (shhs) and WSC (wsc) for training; MrOS (mros, test-only) and the
  census-balanced 500-night set of Jones et al. (2024) (ccshs, cfs, chat,
  mesa, wsc) for external testing (`src/wav2sleep/data/nsrr.py:9-17`;
  split counts: shhs 6441/500/1000, wsc 1380/250/500, chat 1132/100/200,
  mesa 1541/100/200, cfs 452/50/100, ccshs 272/50/100, mros 1000 test,
  census 500/500).
- Labels: AASM 4-class (W, N1+N2, N3, REM); N4 merged into N3,
  MVT/unscored ignored via `ignore_index=-1`
  (`src/wav2sleep/data/utils.py:27-38`, `data/dataset.py:175-176`).
- A stage label at time *t* covers the *preceding* 30 s: labels are shifted
  +30 s at parse time (`src/wav2sleep/data/xml.py:58-60`).
- Recordings are truncated to 10 h (`TRAINING_LENGTH_HOURS`,
  `src/wav2sleep/settings.py:13`).

## Inputs and preprocessing

- Six channels: ABD, THX (256 samples/epoch), ECG, PPG (1024), EOG-L/EOG-R
  (4096) — target rates ≈ 8.53 / 34.13 / 136.53 Hz
  (`src/wav2sleep/settings.py:16-26`).
- ECG/EOG converted to mV; ABD/THX/PPG normalised to [-1, 1] from the header
  physical range (`src/wav2signals/data/edf.py:262-272`).
- Resampling is time-based linear interpolation on the fixed
  samples-per-epoch grid; interior gaps interpolated, leading/trailing gaps
  filled with 0.0 (`src/wav2sleep/data/preprocessing.py:32-52`,
  `data/utils.py:62-98`).
- Normalisation: whole-recording z-score, `std` floored at 1e-6
  (`src/wav2sleep/data/dataset.py:76-87`), unless the causal EMA variant is
  used (`data/normalization.py:106-233`).
- Missing signals are represented by all-`-inf` tensors end-to-end (the
  transformer attention padding mask), not zero-padded
  (`src/wav2sleep/models/wav2sleep.py:149-154`).

## Loss, optimiser, schedule

- Cross-entropy with `ignore_index=-1` (PyTorch Lightning `Task` setup in
  `src/wav2sleep/trainer/main.py`).
- Optimiser/schedule: Adam with cosine-decay warmup as configured under
  `src/wav2sleep/scripts/config/` (see the authors' repo; the shipped
  training config is `scripts/config/training/main.yaml`).
- Batch size: per-subject whole nights, batched as 1-D `[T]` tensors
  (`src/wav2sleep/data/dataset.py:132-183`).

## Augmentation and masking

- Per-batch random polarity inversion with p = 0.5, **training only**
  (`src/wav2sleep/trainer/main.py:131-138,342-353`).
- Channel masking: per-channel Bernoulli keep — ABD 0.7, THX 0.7, ECG 0.5,
  PPG 0.1 — with `[ECG, PPG]` guaranteed as backup channels
  (`src/wav2sleep/trainer/masker.py`,
  `scripts/config/inputs/cardiorespiratory/all.yaml`).

## Hardware / time

Not stated in the repository; training runs on a single GPU in the authors'
setup (PyTorch Lightning), see `src/wav2sleep/scripts/`.
