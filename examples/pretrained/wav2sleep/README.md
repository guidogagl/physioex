# wav2sleep (Carter & Tarassenko, 2024)

Sleep staging from cardiorespiratory signals — ECG, PPG and respiratory effort
(ABD, THX) — using a per-signal CNN encoder + multi-modal attention + temporal
CNN. 4 classes: W, Light (N1+N2), Deep (N3), REM.

**Paper:** arXiv:2411.04644 · **Code:** https://github.com/joncarter1/wav2sleep
· **Weights:** HuggingFace `joncarter/wav2sleep` (also `joncarter/wav2sleep-eog`)

**License of the weights:** MIT.

## Quickstart

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("wav2sleep", verbose=True)
logits = model({"ECG": x, "PPG": torch.full_like(x, float("-inf"))})  # (B, L, 4)
```

`python quickstart.py` loads the model (downloading from HF on first use) and
runs one forward pass on random input.

## Verifying the published numbers

`verify.py` evaluates the model on the test splits of the NSRR datasets of
Carter & Tarassenko's Table 6, per input condition (`all`, `ecg_thx`, `ecg`,
`ppg`), and pools the census-balanced 500-night set of Jones et al. (2024):

```bash
# Slurm note: run on the cluster where the NSRR data live (Sofia), with the
# data/cache env vars from the login profile.
sbatch verify.sbatch        # or
python verify.py --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
python verify.py --max-nights 2     # smoke run (PASS/FAIL reported as REPORT)
```

Results are printed as a table and written to `./verify_results.json`.
Datasets whose raw data is missing are reported as SKIP (never a crash);
`census` rows SKIP when any of the datasets they require is absent.

**Missing data:** the census set includes CFS and CHAT recordings that are not
distributed through the usual NSRR channels and may be absent locally; the
corresponding rows are then SKIPped.

See `TRAINING.md` for how the authors trained the model (physioex does not
ship a training script for it).
