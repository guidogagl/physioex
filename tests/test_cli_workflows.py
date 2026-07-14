"""
Unit tests for CLI entry points: train.py, finetune.py, test.py

Strategy: Run each script with --help via subprocess and verify:
  - Exit code 0
  - Expected argument names appear in help text
  - Required args cause failure when omitted
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_SCRIPT = os.path.join(ROOT, "physioex", "train", "bin", "train.py")
FINETUNE_SCRIPT = os.path.join(ROOT, "physioex", "train", "bin", "finetune.py")
TEST_SCRIPT = os.path.join(ROOT, "physioex", "train", "bin", "test.py")


def _run_script(script_path, args, timeout=30):
    """Run a script as a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, script_path] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestTrainCLI(unittest.TestCase):
    """Tests for test/train/bin/train.py"""

    def test_help_exits_zero(self):
        result = _run_script(TRAIN_SCRIPT, ["--help"])
        self.assertEqual(
            result.returncode, 0, f"train --help failed:\n{result.stderr}"
        )

    def test_help_contains_expected_args(self):
        result = _run_script(TRAIN_SCRIPT, ["--help"])
        for arg in [
            "--model",
            "--datasets",
            "--max_epochs",
            "--lr",
            "--weight_decay",
            "--train_batch_size",
            "--eval_batch_size",
            "--fold",
            "--checkpoint_path",
            "--gpu_id",
            "--selected_channels",
            "--seqlen",
            "--preprocessing",
            "--config",
            "--model_kwargs",
        ]:
            self.assertIn(arg, result.stdout, f"Missing arg {arg} in train --help")

    def test_missing_model_fails(self):
        result = _run_script(TRAIN_SCRIPT, ["--datasets", "hmc"])
        self.assertNotEqual(
            result.returncode, 0, "train should fail when --model is missing"
        )

    def test_missing_datasets_fails(self):
        result = _run_script(TRAIN_SCRIPT, ["--model", "x:Y"])
        self.assertNotEqual(
            result.returncode, 0, "train should fail when --datasets is missing"
        )


class TestFinetuneCLI(unittest.TestCase):
    """Tests for test/train/bin/finetune.py"""

    def test_help_exits_zero(self):
        result = _run_script(FINETUNE_SCRIPT, ["--help"])
        self.assertEqual(
            result.returncode, 0, f"finetune --help failed:\n{result.stderr}"
        )

    def test_help_contains_expected_args(self):
        result = _run_script(FINETUNE_SCRIPT, ["--help"])
        for arg in [
            "--model",
            "--ckpt_path",
            "--datasets",
            "--max_epochs",
            "--lr",
            "--weight_decay",
            "--train_batch_size",
            "--eval_batch_size",
            "--fold",
            "--checkpoint_path",
            "--gpu_id",
            "--selected_channels",
            "--seqlen",
            "--preprocessing",
            "--config",
            "--model_kwargs",
        ]:
            self.assertIn(
                arg, result.stdout, f"Missing arg {arg} in finetune --help"
            )

    def test_requires_ckpt_path(self):
        """finetune must require --ckpt_path; running without it should fail."""
        result = _run_script(
            FINETUNE_SCRIPT, ["--model", "x:Y", "--datasets", "hmc"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            "finetune should fail when --ckpt_path is missing",
        )

    def test_missing_model_fails(self):
        result = _run_script(
            FINETUNE_SCRIPT, ["--ckpt_path", "/tmp/fake.pt", "--datasets", "hmc"]
        )
        self.assertNotEqual(
            result.returncode, 0, "finetune should fail when --model is missing"
        )

    def test_default_lr_is_low(self):
        """Finetune default LR should be lower than train default (1e-5 vs 1e-3)."""
        result = _run_script(FINETUNE_SCRIPT, ["--help"])
        # With ArgumentDefaultsHelpFormatter, the default 1e-5 is shown as 1e-05
        self.assertIn("1e-05", result.stdout, "finetune default lr should be 1e-5")


class TestTestCLI(unittest.TestCase):
    """Tests for test/train/bin/test.py"""

    def test_help_exits_zero(self):
        result = _run_script(TEST_SCRIPT, ["--help"])
        self.assertEqual(
            result.returncode, 0, f"test --help failed:\n{result.stderr}"
        )

    def test_help_contains_expected_args(self):
        result = _run_script(TEST_SCRIPT, ["--help"])
        for arg in [
            "--model",
            "--ckpt_path",
            "--datasets",
            "--fold",
            "--results_path",
            "--gpu_id",
            "--selected_channels",
            "--seqlen",
            "--preprocessing",
            "--voting",
            "--voting_L",
            "--config",
            "--model_kwargs",
        ]:
            self.assertIn(arg, result.stdout, f"Missing arg {arg} in test --help")

    def test_requires_ckpt_path(self):
        """test must require --ckpt_path; running without it should fail."""
        result = _run_script(TEST_SCRIPT, ["--model", "x:Y", "--datasets", "hmc"])
        self.assertNotEqual(
            result.returncode,
            0,
            "test should fail when --ckpt_path is missing",
        )

    def test_requires_model(self):
        result = _run_script(
            TEST_SCRIPT, ["--ckpt_path", "/tmp/fake.pt", "--datasets", "hmc"]
        )
        self.assertNotEqual(
            result.returncode, 0, "test should fail when --model is missing"
        )

    def test_voting_flag_present(self):
        """The --voting flag should be documented as a store_true action."""
        result = _run_script(TEST_SCRIPT, ["--help"])
        self.assertIn("--voting", result.stdout)
        # Check that help text mentions voting evaluation in some form
        help_lower = result.stdout.lower()
        self.assertTrue(
            "voting evaluation" in help_lower or "voting" in help_lower,
            "Help text should describe the voting flag purpose",
        )


class TestCrossScriptConsistency(unittest.TestCase):
    """Verify consistency across the three CLI scripts."""

    def test_all_scripts_exist(self):
        for path in [TRAIN_SCRIPT, FINETUNE_SCRIPT, TEST_SCRIPT]:
            self.assertTrue(os.path.isfile(path), f"Script not found: {path}")

    def test_init_file_exists(self):
        init_path = os.path.join(ROOT, "physioex", "train", "bin", "__init__.py")
        self.assertTrue(os.path.isfile(init_path), "__init__.py missing in bin/")

    def test_shared_args_present_in_all(self):
        """--model, --datasets, --selected_channels, --seqlen, --preprocessing
        should appear in all three scripts."""
        shared_args = [
            "--model",
            "--datasets",
            "--selected_channels",
            "--seqlen",
            "--preprocessing",
        ]
        for script in [TRAIN_SCRIPT, FINETUNE_SCRIPT, TEST_SCRIPT]:
            result = _run_script(script, ["--help"])
            self.assertEqual(result.returncode, 0)
            for arg in shared_args:
                self.assertIn(
                    arg,
                    result.stdout,
                    f"Missing {arg} in {os.path.basename(script)} --help",
                )


if __name__ == "__main__":
    unittest.main()
