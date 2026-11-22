# Jones, Itti & Sheth (2024) — ECG-only sleep staging

Sleep staging from a single ECG channel: per-epoch CNN front-end (31-tap
convolutions, avg+max pooling), a temporal conv net over the epochs, and
auxiliary per-epoch features (sex, age, night, midnight offset). 5 AASM
classes: W, N1, N2, N3, REM.

**Paper:** Computers in Biology and Medicine (2024),
doi:10.1016/j.compbiomed.2024.108545 ·
**Code/weights:** https://github.com/adammj/ecg-sleep-staging

**License of the weights:** AGPL-3.0 (copyleft — may propagate to derivative
works). `load_from_pretrained` warns about this on every load.

Seven released checkpoints are available as
`jones-ecg-primary` (default, also `jones-ecg`), `jones-ecg-realtime`,
`jones-ecg-no-demographics`, `jones-ecg-no-time`, `jones-ecg-only-ecg`,
`jones-ecg-cyclic`, `jones-ecg-cyclic-realtime`.

## Quickstart

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("jones-ecg", verbose=True)
pred = model.predict_night({"ECG": x}, {"id": "demo"})   # (L,) class indices
```

`python quickstart.py` loads the primary checkpoint (downloading on first
use) and runs one pass on random input.

## Verifying the published numbers

`verify.py` evaluates the chosen `--variant` on the census-balanced 500-night
test set of Jones et al. (2024) (the `wav2sleep-census` split, spread over
ccshs, cfs, chat, mesa, wsc) and on the per-dataset test sets:

```bash
# Slurm note: run on the cluster where the NSRR data live (Sofia), with the
# data/cache env vars from the login profile.
python verify.py --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
python verify.py --variant realtime
python verify.py --max-nights 2    # smoke run
```

The headline is the **median per-recording kappa** (0.725 in the paper).
`reference_per_record.csv` contains the authors' own per-recording kappas;
`verify.py` prints the median of those over exactly the recordings it
evaluated, next to ours — the like-for-like comparison.  Results are written
to `./verify_results.json` (rows, per-recording kappas, run info).

**Missing data:** the census set includes CFS and CHAT recordings that are
not distributed through the usual NSRR channels; when they are absent
locally the affected datasets (and the census row) are reported as SKIP
(never a crash), and the like-for-like reference median covers only the
recordings actually evaluated.

See `TRAINING.md` for the authors' training recipe (physioex does not ship a
training script for this model).
