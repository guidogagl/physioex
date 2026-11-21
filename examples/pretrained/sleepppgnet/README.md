# SleepPPG-Net (Kotzen et al., 2023) — replica weights

Sleep staging from a single PPG (plethysmogram) channel: a 1-D CNN window
encoder + time-distributed dense + 2 dilated conv layers. 4 classes:
W, Light (N1+N2), Deep (N3), REM.

**Paper (architecture):** Kotzen et al. (2023), IEEE JBHI, arXiv:2202.05735.
**Weights:** **third-party replica** (Franz Ehrlich,
`sleep-is-all-you-need/SPPPPG-Net` on HuggingFace), retrained with the
SleePyPhases framework — *not* the authors' model.

**License of the weights:** fair non-commercial research use only, per the
SPPPPG-Net repository. `load_from_pretrained` warns about this on every load.

## Quickstart

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("sleepppgnet-replica", verbose=True)
logits = model(torch.randn(1, 1200, 1024))   # (1, 1200, 4)
```

`python quickstart.py` loads the model (downloading on first use) and runs one
forward pass on random input.

## Verifying the published number

`verify.py` evaluates the replica on the 204 MESA test recordings of its
published split (`split="sleepppgnet-replica"`):

```bash
# Slurm note: run on the cluster where the MESA data live (Sofia), with the
# data/cache env vars from the login profile.
python verify.py --data-root $PHYSIOEX_DATA --cache-dir $PHYSIOEX_CACHE_DIR
python verify.py --max-nights 2    # smoke run
```

The paper's headline is the median per-patient kappa 0.75 (IQR 0.69–0.81).
Because the weights are a third-party re-training, the row is *reported* next
to the paper value, not asserted (`assert: false` in `paper_metrics.json`);
per-recording kappas are written to `./verify_results.json`.

Datasets whose raw data is missing are reported as SKIP (never a crash).  The
MESA test set of this split is a fixed list of 204 recordings; only the ones
present locally are evaluated.

See `TRAINING.md` for the authors'/replica's training recipe (physioex does
not ship a training script for this model).
