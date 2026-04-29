"""
Unit tests for P3-B8 (regression metrics), P3-C2 (no hardcoded paths),
and P3-C3 (pyproject.toml deps).

Run:
    cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_metrics_and_packaging.py
"""

import sys
import math
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

import torch
from physioex.train.metrics import mse_score, mae_score, r2_score, REGRESSION_METRICS, CLASSIFICATION_METRICS

# ── helpers ────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ══════════════════════════════════════════════════════════════════════════
# P3-B8: Regression metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n=== P3-B8: Regression metrics ===")

# 1. mse_score with perfect predictions
targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
outputs = targets.clone()
val = mse_score(outputs, targets, ignore_index=None)
_check("mse perfect predictions == 0", val == 0.0, f"got {val}")

# 2. mse_score with constant offset +2
outputs_off = targets + 2.0
val = mse_score(outputs_off, targets, ignore_index=None)
_check("mse constant offset +2 == 4.0", math.isclose(val, 4.0, rel_tol=1e-5), f"got {val}")

# 3. mae_score with constant offset +2
val = mae_score(outputs_off, targets, ignore_index=None)
_check("mae constant offset +2 == 2.0", math.isclose(val, 2.0, rel_tol=1e-5), f"got {val}")

# 4. r2_score with perfect predictions (with variance)
targets_var = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
outputs_var = targets_var.clone()
val = r2_score(outputs_var, targets_var, ignore_index=None)
_check("r2 perfect predictions close to 1.0", math.isclose(val, 1.0, rel_tol=1e-5), f"got {val}")

# 5. ignore_index masking -- half targets are -1
targets_masked = torch.tensor([1.0, -1.0, 3.0, -1.0, 5.0])
outputs_masked = torch.tensor([1.0, 999.0, 3.0, 999.0, 5.0])
val = mse_score(outputs_masked, targets_masked, ignore_index=-1)
_check("mse with ignore_index masking == 0", val == 0.0, f"got {val}")

val = mae_score(outputs_masked, targets_masked, ignore_index=-1)
_check("mae with ignore_index masking == 0", val == 0.0, f"got {val}")

# 6. Empty after masking -- all targets are -1
targets_all_masked = torch.tensor([-1.0, -1.0, -1.0])
outputs_all_masked = torch.tensor([10.0, 20.0, 30.0])
val_mse = mse_score(outputs_all_masked, targets_all_masked, ignore_index=-1)
val_mae = mae_score(outputs_all_masked, targets_all_masked, ignore_index=-1)
val_r2 = r2_score(outputs_all_masked, targets_all_masked, ignore_index=-1)
_check("mse all masked returns 0.0", val_mse == 0.0, f"got {val_mse}")
_check("mae all masked returns 0.0", val_mae == 0.0, f"got {val_mae}")
_check("r2 all masked returns 0.0", val_r2 == 0.0, f"got {val_r2}")

# Verify the dicts contain the right keys
_check("REGRESSION_METRICS has mse/mae/r2",
       set(REGRESSION_METRICS.keys()) == {"mse", "mae", "r2"})
_check("CLASSIFICATION_METRICS has expected keys",
       set(CLASSIFICATION_METRICS.keys()) == {
           "accuracy", "f1_score", "precision", "recall",
           "cohen_kappa", "confusion_matrix", "support"})

# ══════════════════════════════════════════════════════════════════════════
# P3-C2: No hardcoded /home/dev paths in test/examples/ .py files
# ══════════════════════════════════════════════════════════════════════════
print("\n=== P3-C2: No hardcoded absolute paths ===")

examples_dir = _REPO / "test" / "examples"
result = subprocess.run(
    ["grep", "-rn", "/home/dev/physioex", "--include=*.py", str(examples_dir)],
    capture_output=True,
    text=True,
)
matches = result.stdout.strip()
_check("No /home/dev/physioex in test/examples/*.py", matches == "",
       f"found:\n{matches}" if matches else "")

# ══════════════════════════════════════════════════════════════════════════
# P3-C3: pyproject.toml dependencies
# ══════════════════════════════════════════════════════════════════════════
print("\n=== P3-C3: pyproject.toml deps ===")

try:
    import tomllib
except ImportError:
    import tomli as tomllib

pyproject_path = _REPO / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    data = tomllib.load(f)

deps = data["project"]["dependencies"]
for pkg in ["rich", "nvitop", "psutil", "torchinfo"]:
    found = any(pkg in d for d in deps)
    _check(f"{pkg} in dependencies", found)

# Also verify existing deps were NOT removed
for existing in ["torch", "torchmetrics", "captum", "lightning"]:
    found = any(existing in d for d in deps)
    _check(f"existing dep {existing} still present", found)

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results: {_passed} passed, {_failed} failed, {_passed + _failed} total")
if _failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
