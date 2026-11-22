# How SleepPPG-Net was trained

Paper (architecture): Kotzen, C., Mollenhauer, M., Schirrmacher, T., &
Schirrmeister, G. R. (2023). *SleepPPG-Net: A Deep Learning Model for Sleep
Staging using Pulse Volume Recording Signals.* IEEE J. Biomed. Health Inform.
27(4):924-932. arXiv:2202.05735.

**The weights shipped with physioex are a third-party replica** (Franz Ehrlich,
`sleep-is-all-you-need/SPPPPG-Net`, fair non-commercial research license), not
the authors' model; the training recipe below is the one of that replica
(SleePyPhases / pyPhasesML framework), so it may differ from the paper's.
Physioex does not ship a training script for these models.

## Data and split

- MESA plethysmogram (the `mesa` loader of the SleePyPhases framework),
  annotation `SleepStagesAASM` (`src/config.yaml:28-29,39-40`).
- 2056 unique records; the 204-record MESA test set is the first 204 entries of
  the hand-ordered `recordIds` list (`src/config.yaml:174-190`), the rest
  shuffled with `random.seed(2025)` for trainval
  (`src/phases/Init.py:14-27`).

## Inputs and preprocessing

Per record (`recordWise: True`), in this order
(`src/SignalPreprocessing.py`, `src/config.yaml:36-37`):

1. 8th-order Chebyshev type-II lowpass, 8 Hz cutoff, -40 dB stop band,
   zero-phase (`SignalPreprocessing.py:27-53`).
2. FFT resampling (`scipy.signal.resample`) to 1024 samples per 30 s
   (34.133… Hz) (`SignalPreprocessing.py:9-14`).
3. Clip to mean ± 3·std over the whole night, then z-score with the
   **clipped** statistics (`SignalPreprocessing.py:55-66`).
4. Pad/truncate to 10 h (1 228 800 samples, 1200 epochs); labels padded with
   −1 (`src/DataManipulation.py:55-82`, `config.yaml:49-64`).

## Labels

AASM codes; N1/N2 collapsed to Light and N3 to Deep (`toLightDeepSleep`,
`src/DataManipulation.py:11-16`) → 4 classes [W, Light, Deep, REM]
(model order in the replica: [Wake, REM, Light, Deep]).  −1-padded epochs are
ignored in the loss (`ignore_index=-1`, `src/SleepPPGNet.py:413`).

## Loss, optimiser, schedule

- Loss: plain `crossentropy` (`config.yaml:18-20,106`); the temporal-balance
  variants exist in the code but are not the active loss
  (`src/SleepPPGNet.py:247-365`).
- Optimiser: Adam (`config.yaml:104`, `optimizer: adams` is a typo/alias),
  lr 2.5e-4, decay 1e-3, batch size 8 (`config.yaml:104-112`).
- Early stopping after 25 epochs without improvement of val f1/kappa, max
  1000 epochs (`config.yaml:109-112`).

## Hardware / time

Not stated in the package.
