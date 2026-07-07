"""
Unit tests for 4 trainer enhancements:
  - P2-B6:  seed_everything + seed parameter on train/evaluate/voting_evaluate
  - P2-D6-Bug4: multidevicetrainer _run_epoch is @classmethod (not @staticmethod)
  - P2-D6-Bug3: multidevicetrainer uses the pluggable Logger (build_logger)
  - P3-D4:  gradient accumulation (accumulate_grad_batches parameter)
  - P3-D5:  early stopping (early_stopping_patience parameter)

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_trainer_enhancements.py
"""

import os
import sys
import inspect
import re
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from physioex.train.trainer import Trainer, seed_everything

passed, failed = 0, 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# ---------------------------------------------------------------------------
# Test 1: seed_everything is reproducible
# ---------------------------------------------------------------------------
def test_seed_reproducibility():
    try:
        import random
        import numpy as np

        seed_everything(42)
        py_vals_1 = [random.random() for _ in range(5)]
        np_vals_1 = np.random.rand(5).tolist()
        torch_vals_1 = torch.rand(5).tolist()

        seed_everything(42)
        py_vals_2 = [random.random() for _ in range(5)]
        np_vals_2 = np.random.rand(5).tolist()
        torch_vals_2 = torch.rand(5).tolist()

        assert py_vals_1 == py_vals_2, f"Python random mismatch: {py_vals_1} vs {py_vals_2}"
        assert np_vals_1 == np_vals_2, f"NumPy random mismatch"
        assert torch_vals_1 == torch_vals_2, f"Torch random mismatch"
        report("P2-B6: seed_everything produces reproducible sequences", True)
    except Exception as exc:
        report("P2-B6: seed_everything produces reproducible sequences", False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: Trainer.train accepts seed parameter (default 42)
# ---------------------------------------------------------------------------
def test_train_has_seed_param():
    try:
        sig = inspect.signature(Trainer.train)
        assert "seed" in sig.parameters, "seed parameter missing from Trainer.train"
        default = sig.parameters["seed"].default
        assert default == 42, f"seed default is {default}, expected 42"
        report("P2-B6: Trainer.train has seed parameter (default=42)", True)
    except Exception as exc:
        report("P2-B6: Trainer.train has seed parameter (default=42)", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: Trainer.train accepts accumulate_grad_batches parameter (default 1)
# ---------------------------------------------------------------------------
def test_train_has_accumulate_param():
    try:
        sig = inspect.signature(Trainer.train)
        assert "accumulate_grad_batches" in sig.parameters, "accumulate_grad_batches missing from Trainer.train"
        default = sig.parameters["accumulate_grad_batches"].default
        assert default == 1, f"accumulate_grad_batches default is {default}, expected 1"
        report("P3-D4: Trainer.train has accumulate_grad_batches parameter (default=1)", True)
    except Exception as exc:
        report("P3-D4: Trainer.train has accumulate_grad_batches parameter (default=1)", False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: Trainer.train accepts early_stopping_patience parameter (default None)
# ---------------------------------------------------------------------------
def test_train_has_early_stopping_param():
    try:
        sig = inspect.signature(Trainer.train)
        assert "early_stopping_patience" in sig.parameters, "early_stopping_patience missing from Trainer.train"
        default = sig.parameters["early_stopping_patience"].default
        assert default is None, f"early_stopping_patience default is {default}, expected None"
        report("P3-D5: Trainer.train has early_stopping_patience parameter (default=None)", True)
    except Exception as exc:
        report("P3-D5: Trainer.train has early_stopping_patience parameter (default=None)", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: Gradient accumulation -- optimizer.step() called half as often
# ---------------------------------------------------------------------------
def test_gradient_accumulation_behavior():
    try:
        N_CLASSES = 3

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, N_CLASSES)

            def forward(self, x):
                # x: (batch, seq_len, 4) -> (batch, seq_len, N_CLASSES)
                b, s = x.shape[0], x.shape[1]
                out = self.fc(x.reshape(b * s, -1))
                return out.reshape(b, s, N_CLASSES)

        model = TinyModel()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Count optimizer.step() calls
        real_step = optimizer.step
        step_count = [0]

        def counting_step(*args, **kwargs):
            step_count[0] += 1
            return real_step(*args, **kwargs)

        optimizer.step = counting_step

        total_steps = 10
        accumulate = 2

        for step_idx in range(total_steps):
            # batch: (inputs, targets) matching the shape conventions
            # inputs: (batch, seq_len, features), targets: (batch, seq_len)
            inputs = torch.randn(2, 3, 4)
            targets = torch.randint(0, N_CLASSES, (2, 3))
            Trainer._train_step(
                model=model,
                batch=(inputs, targets),
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=torch.device("cpu"),
                step=step_idx,
                accumulate_grad_batches=accumulate,
            )

        expected_steps = total_steps // accumulate  # 10 / 2 = 5
        assert step_count[0] == expected_steps, (
            f"optimizer.step() called {step_count[0]} times, expected {expected_steps}"
        )
        report("P3-D4: gradient accumulation halves optimizer.step() calls", True)
    except Exception as exc:
        report("P3-D4: gradient accumulation halves optimizer.step() calls", False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: multidevice _run_epoch is @classmethod
# ---------------------------------------------------------------------------
def test_multidevice_run_epoch_is_classmethod():
    try:
        from physioex.train.multidevicetrainer import Trainer as MultiTrainer
        # A classmethod has __func__ attribute
        assert hasattr(MultiTrainer.__dict__["_run_epoch"], "__func__"), (
            "_run_epoch does not have __func__ -- it is not a classmethod"
        )
        report("P2-D6-Bug4: multidevice _run_epoch is @classmethod", True)
    except Exception as exc:
        report("P2-D6-Bug4: multidevice _run_epoch is @classmethod", False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: multidevice trainer uses the pluggable Logger (build_logger)
# ---------------------------------------------------------------------------
def test_multidevice_uses_logger():
    try:
        import physioex.train.multidevicetrainer as mdt

        with open(mdt.__file__, "r") as f:
            source = f.read()

        # The CSV/matplotlib LossTracker was decommissioned in favour of the
        # pluggable Logger abstraction (TensorBoard / W&B).
        assert "LossTracker" not in source, (
            "multidevicetrainer.py still references the removed LossTracker"
        )
        has_logger = bool(re.search(r"build_logger", source))
        assert has_logger, "multidevicetrainer.py does not use build_logger"
        report("P2-D6-Bug3: multidevice trainer uses build_logger", True)
    except Exception as exc:
        report("P2-D6-Bug3: multidevice trainer uses build_logger", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running trainer enhancement tests")
    print("=" * 60)

    test_seed_reproducibility()
    test_train_has_seed_param()
    test_train_has_accumulate_param()
    test_train_has_early_stopping_param()
    test_gradient_accumulation_behavior()
    test_multidevice_run_epoch_is_classmethod()
    test_multidevice_uses_logger()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
