# How the Jones et al. (2024) ECG model was trained

Authors: Jones, A. M., Itti, L. & Sheth, B. R. (2024). *Expert-level sleep
staging using an electrocardiography-only feed-forward neural network.*
Computers in Biology and Medicine, doi:10.1016/j.compbiomed.2024.108545 —
code: https://github.com/adammj/ecg-sleep-staging

Physioex does not ship a training script for these models; the references
below point into the authors' repository.

## Data and split

- ECG-only recordings from CCSHS, CFS, CHAT, MESA, MrOS, SHHS and WSC;
  benchmark of 4,000 recordings (`benchmark_dataset/benchmark_v1.0.xlsx`),
  train/val/test built by `paper/5_sets/build_sets.m` (MATLAB `rng(12)`,
  stratified by sex × decade, subject-disjoint, dataset-ratio balanced;
  `build_sets.m:46,160-198`).
- Test set: 500 recordings / 571,141 scored epochs (`README.md:156,162`).

## Inputs and preprocessing (MATLAB, `paper/0_data_processing/`)

1. Two-channel lead subtraction (`channel_1 - channel_2`) for polarity
   (`2_cut_clip_subtract_resample/cut_to_resample_process.m:207-211`).
2. Loose-lead clipping: 0.5-clip-ratio moving average, 1 s gap fill,
   per-channel median removed without adding it back
   (`remove_large_clips.m:17-53`).
3. Baseline + line noise: Chebychev-2 high-pass 0.5 Hz, 60 Hz (or aliased
   `fs−60`) notch, iterative removal of periodic noise ≥ 4 Hz
   (`cut_to_resample_process.m:232-264`).
4. Resample to 256 Hz (MATLAB `resample`) → 7680 samples per 30-s epoch
   (`cut_to_resample_process.m:272-283`).
5. Per-recording R-wave template (local-variation peaks → archetypes →
   median template) and normalisation factor = 2 × P90 of the per-beat
   max absolute value (`paper/3_templates/find_norm_factor.m:67-73`); the
   ECG is divided by that factor and hard-clipped to ±1
   (`paper/5_sets/create_set_files.m:117-127`).
6. Auxiliary per-epoch features: [sex, age/100, night, midnight offset]
   (`README.md:87-95`).

## Labels

5 classes W, N1, N2, N3 (incl. S4), REM; unscored → Wake with weight 0
(`create_set_files.m:52-86,106-114`).  No wake trimming.

## Loss, optimiser, schedule

- Loss: per-stage kappa product — `1 − prod(kappas)^(1/stage_count)` on the
  (weight-scaled) continuous confusion (`sleeploss.py:34-75`).
- Optimiser: AdamS (stable weight decay), lr 1e-3, weight decay 5e-4
  (`adams.py:10-141`, `train.py:315-321`);
  `ReduceLROnPlateau(mode='max', factor=0.5, patience=50)` on the val
  overall-kappa (`train.py:329-335`).
- Batch size 10, seed 12, deterministic mode (`train.py:23,60-78`).

## Augmentation

- 50 % of training nights have the ECG polarity flipped
  (`sleepdataset.py:368-370`, `train_params.json: train_invert_ecg`).

## Hardware / time

Not stated in the repository; the released checkpoints are named by epoch and
val kappa, e.g. `epoch_0647_kappa_0.724.pt` (primary).
