"""
Unit tests for the 4 P0 trainer bug fixes:
  - A1:  _get_parameters_from_config NameError fix
  - B7:  voting_evaluate sliding-window method
  - D6-Bug1: multidevicetrainer _train_step unpacking (4 values)
  - D6-Bug2: multidevicetrainer _eval_step unpacking (3 values)

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_trainer_fixes.py
"""

import os
import sys
import tempfile
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from physioex.train.trainer import _get_parameters_from_config, Trainer
import torch
from torch.utils.data import Dataset, DataLoader

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
# Test 1: A1 -- _get_parameters_from_config handles missing YAML
# ---------------------------------------------------------------------------
def test_config_missing_yaml():
    orig_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = _get_parameters_from_config()
            # Must be a dict with all-None values, no NameError
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            for k, v in result.items():
                assert v is None, f"Key {k!r} should be None, got {v!r}"
            report("A1: missing YAML returns all-None dict", True)
    except Exception as exc:
        report("A1: missing YAML returns all-None dict", False, str(exc))
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Test 2: A1 -- _get_parameters_from_config reads existing YAML
# ---------------------------------------------------------------------------
def test_config_reads_yaml():
    orig_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "PHYSIOEX_CONFIG.yaml")
            with open(yaml_path, "w") as f:
                f.write("Trainer:\n  max_epochs: 42\n  train_batch_size: 64\n")
            os.chdir(tmpdir)
            result = _get_parameters_from_config()
            assert result["max_epochs"] == 42, f"max_epochs={result['max_epochs']}"
            assert result["train_batch_size"] == 64, f"train_batch_size={result['train_batch_size']}"
            report("A1: reads existing YAML correctly", True)
    except Exception as exc:
        report("A1: reads existing YAML correctly", False, str(exc))
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Test 3: A1 -- _get_parameters_from_config handles YAML without Trainer key
# ---------------------------------------------------------------------------
def test_config_no_trainer_section():
    orig_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "PHYSIOEX_CONFIG.yaml")
            with open(yaml_path, "w") as f:
                f.write("PhysioExDataset:\n  datasets:\n    - hmc\n")
            os.chdir(tmpdir)
            result = _get_parameters_from_config()
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            for k, v in result.items():
                assert v is None, f"Key {k!r} should be None, got {v!r}"
            report("A1: YAML without Trainer section returns all-None dict", True)
    except Exception as exc:
        report("A1: YAML without Trainer section returns all-None dict", False, str(exc))
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Test 4: B7 -- voting_evaluate produces correct output shape
# ---------------------------------------------------------------------------
def test_voting_evaluate():
    try:
        N_CLASSES = 5
        FEATURES = 16
        NIGHT_LEN = 50
        L = 10

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(FEATURES, N_CLASSES)

            def forward(self, x):
                # x: (batch, L, features) -> (batch, L, n_classes)
                b, seq = x.shape[0], x.shape[1]
                out = self.fc(x.reshape(b * seq, -1))
                return out.reshape(b, seq, N_CLASSES)

        class FakeNightDataset(Dataset):
            """Yields full-night samples: (night_length, features) and (night_length,) targets."""
            def __init__(self, n_subjects=3):
                self.n = n_subjects

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                inputs = torch.randn(NIGHT_LEN, FEATURES)
                targets = torch.randint(0, N_CLASSES, (NIGHT_LEN,))
                return inputs, targets

        model = TinyModel()
        loader = DataLoader(FakeNightDataset(n_subjects=3), batch_size=1, shuffle=False)

        # Change CWD to a temp dir so _get_parameters_from_config doesn't pick up
        # real PHYSIOEX_CONFIG.yaml
        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            results = Trainer.voting_evaluate(
                model=model,
                dataset=loader,
                L=L,
                gpu_id=None,
            )
        os.chdir(orig_cwd)

        expected_keys = {"accuracy", "f1_score", "precision", "recall", "cohen_kappa", "confusion_matrix", "support"}
        assert set(results.keys()) == expected_keys, f"Keys mismatch: {set(results.keys())} vs {expected_keys}"
        acc = results["accuracy"]
        assert isinstance(acc, float), f"accuracy type={type(acc)}"
        assert 0.0 <= acc <= 1.0, f"accuracy={acc} out of [0,1]"
        report("B7: voting_evaluate produces correct output", True)
    except Exception as exc:
        report("B7: voting_evaluate produces correct output", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: D6-Bug1 -- multidevicetrainer _train_step unpacks 4 values
# ---------------------------------------------------------------------------
def test_multidevice_train_step_unpacking():
    try:
        src_path = os.path.join(ROOT, "physioex", "train", "multidevicetrainer.py")
        with open(src_path, "r") as f:
            source = f.read()

        # Find all lines that assign from _train_step(
        # Pattern: <names> = <something>._train_step(  or  <names> = _train_step(
        # We look for tuple-unpacking assignments to _train_step
        pattern = r"(\w+(?:\s*,\s*\w+)*)\s*=\s*\w+\._train_step\("
        matches = re.findall(pattern, source)
        assert len(matches) > 0, "No _train_step assignment found in multidevicetrainer.py"

        for match in matches:
            names = [n.strip() for n in match.split(",")]
            assert len(names) == 4, (
                f"_train_step unpacking has {len(names)} names ({names}), expected 4"
            )

        report("D6-Bug1: _train_step unpacks 4 values in multidevicetrainer.py", True)
    except Exception as exc:
        report("D6-Bug1: _train_step unpacks 4 values in multidevicetrainer.py", False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: D6-Bug2 -- multidevicetrainer _eval_step unpacks 3 values
# ---------------------------------------------------------------------------
def test_multidevice_eval_step_unpacking():
    try:
        src_path = os.path.join(ROOT, "physioex", "train", "multidevicetrainer.py")
        with open(src_path, "r") as f:
            source = f.read()

        pattern = r"(\w+(?:\s*,\s*\w+)*)\s*=\s*\w+\._eval_step\("
        matches = re.findall(pattern, source)
        assert len(matches) > 0, "No _eval_step assignment found in multidevicetrainer.py"

        for match in matches:
            names = [n.strip() for n in match.split(",")]
            assert len(names) == 3, (
                f"_eval_step unpacking has {len(names)} names ({names}), expected 3"
            )

        report("D6-Bug2: _eval_step unpacks 3 values in multidevicetrainer.py", True)
    except Exception as exc:
        report("D6-Bug2: _eval_step unpacks 3 values in multidevicetrainer.py", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running trainer fix tests")
    print("=" * 60)

    test_config_missing_yaml()
    test_config_reads_yaml()
    test_config_no_trainer_section()
    test_voting_evaluate()
    test_multidevice_train_step_unpacking()
    test_multidevice_eval_step_unpacking()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
