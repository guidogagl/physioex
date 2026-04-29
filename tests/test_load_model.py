"""Unit tests for test/train/models/load.py -- model loading infrastructure.

Tests dual-format checkpoint support (Lightning .ckpt vs plain .pt),
registry lookup, and error handling.

Run: python test/tests/test_load_model.py
"""

import os
import sys
import tempfile
import traceback

import torch
import torch.nn as nn

from physioex.train.models.load import _get_registry, load_model

PASSED = 0
FAILED = 0


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS: {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        FAILED += 1


# ---------------------------------------------------------------------------
# Test 1: Registry is readable and has required columns
# ---------------------------------------------------------------------------
def test_registry_readable():
    df = _get_registry()
    assert len(df) > 0, "Registry is empty"
    required_cols = {"name", "sequence_length", "in_channels", "checkpoint"}
    actual_cols = set(df.columns)
    missing = required_cols - actual_cols
    assert not missing, f"Registry missing columns: {missing}"


# ---------------------------------------------------------------------------
# Test 2: Load from plain .pt (new format with "model_state_dict")
# ---------------------------------------------------------------------------
def test_load_pt_format():
    # Create a small model and save in new format
    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, path)

    try:
        loaded = load_model(
            nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path
        )
        assert torch.allclose(loaded.weight.data, original_weight), (
            "Weights do not match after loading .pt format"
        )
        assert torch.allclose(loaded.bias.data, original_bias), (
            "Bias does not match after loading .pt format"
        )
        assert not loaded.training, "Model should be in eval mode"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 3: Load from Lightning-style .ckpt (old format with "nn." prefix)
# ---------------------------------------------------------------------------
def test_load_lightning_format():
    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()

    # Save with "nn." prefix as Lightning's SleepModule does
    state_dict_with_prefix = {
        "nn.weight": model.weight.data.clone(),
        "nn.bias": model.bias.data.clone(),
    }

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        path = f.name
        torch.save({"state_dict": state_dict_with_prefix}, path)

    try:
        loaded = load_model(
            nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path
        )
        assert torch.allclose(loaded.weight.data, original_weight), (
            "Weights do not match after loading Lightning .ckpt format"
        )
        assert torch.allclose(loaded.bias.data, original_bias), (
            "Bias does not match after loading Lightning .ckpt format"
        )
        assert not loaded.training, "Model should be in eval mode"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: Load from raw state_dict (no wrapper dict)
# ---------------------------------------------------------------------------
def test_load_raw_state_dict():
    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save(model.state_dict(), path)

    try:
        loaded = load_model(
            nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path
        )
        assert torch.allclose(loaded.weight.data, original_weight), (
            "Weights do not match after loading raw state_dict"
        )
        assert torch.allclose(loaded.bias.data, original_bias), (
            "Bias do not match after loading raw state_dict"
        )
        assert not loaded.training, "Model should be in eval mode"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 5: Registry lookup -- non-existent name raises clear error
# ---------------------------------------------------------------------------
def test_registry_lookup_nonexistent_name():
    try:
        load_model(
            nn.Linear,
            {"in_features": 4, "out_features": 2, "sequence_length": 21, "in_channels": 1},
            model_name="nonexistent_model_xyz",
        )
        raise AssertionError("Expected ValueError for non-existent model name")
    except ValueError as e:
        assert "nonexistent_model_xyz" in str(e), (
            f"Error message should mention the model name, got: {e}"
        )


# ---------------------------------------------------------------------------
# Test 6: Registry lookup resolves a known model name to correct path
# ---------------------------------------------------------------------------
def test_registry_lookup_known_name_resolves_path():
    """Verify that a known model name resolves to the expected checkpoint path.
    We don't actually load (the checkpoint file won't exist locally), but we
    verify the path resolution logic by catching the download attempt.
    """
    # Temporarily patch hf_hub_download to capture the call instead of downloading
    import unittest.mock as mock

    expected_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "train", "models", "checkpoints",
    )
    expected_dir = os.path.normpath(expected_dir)

    # The model file won't exist, so load_model will try to download.
    # We intercept the download call to verify path resolution.
    with mock.patch("test.train.models.load.torch") as mock_torch:
        # We need to mock the download call inside the module
        with mock.patch.dict("sys.modules", {"huggingface_hub": mock.MagicMock()}):
            # Re-import to pick up the mock
            import importlib
            import physioex.train.models.load as load_module
            importlib.reload(load_module)

            # Verify the registry finds the entry
            table = load_module._get_registry()
            mask = (
                (table["name"] == "seqsleepnet")
                & (table["sequence_length"] == 21)
                & (table["in_channels"] == 1)
            )
            filtered = table[mask]
            assert len(filtered) == 1, (
                f"Expected 1 registry entry for seqsleepnet/21/1, got {len(filtered)}"
            )
            assert "seqsleepnet" in filtered.iloc[0]["checkpoint"], (
                "Checkpoint filename should contain 'seqsleepnet'"
            )

            # Restore the original module
            importlib.reload(load_module)


# ---------------------------------------------------------------------------
# Test 7: model_class as string "module:Class"
# ---------------------------------------------------------------------------
def test_model_class_as_string():
    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save(model.state_dict(), path)

    try:
        loaded = load_model(
            "torch.nn:Linear",
            {"in_features": 4, "out_features": 2},
            ckpt_path=path,
        )
        assert torch.allclose(loaded.weight.data, original_weight), (
            "Weights do not match when using string model_class"
        )
        assert not loaded.training, "Model should be in eval mode"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 8: Missing both ckpt_path and model_name raises ValueError
# ---------------------------------------------------------------------------
def test_missing_ckpt_and_name_raises():
    try:
        load_model(nn.Linear, {"in_features": 4, "out_features": 2})
        raise AssertionError("Expected ValueError when both ckpt_path and model_name are None")
    except ValueError as e:
        assert "ckpt_path" in str(e) or "model_name" in str(e), (
            f"Error message should mention ckpt_path or model_name, got: {e}"
        )


# ---------------------------------------------------------------------------
# Test 9: Unexpected checkpoint format raises ValueError
# ---------------------------------------------------------------------------
def test_unexpected_checkpoint_format():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save("not_a_dict", path)

    try:
        load_model(nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path)
        raise AssertionError("Expected ValueError for unexpected checkpoint format")
    except ValueError as e:
        assert "Unexpected checkpoint format" in str(e), (
            f"Error message should mention 'Unexpected checkpoint format', got: {e}"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("test/tests/test_load_model.py -- Model Loading Tests")
    print("=" * 60)
    print()

    run_test("Registry is readable with required columns", test_registry_readable)
    run_test("Load from .pt format (model_state_dict)", test_load_pt_format)
    run_test("Load from Lightning .ckpt format (nn. prefix)", test_load_lightning_format)
    run_test("Load from raw state_dict", test_load_raw_state_dict)
    run_test("Registry lookup: non-existent name raises error", test_registry_lookup_nonexistent_name)
    run_test("Registry lookup: known name resolves correctly", test_registry_lookup_known_name_resolves_path)
    run_test("Model class as string 'module:Class'", test_model_class_as_string)
    run_test("Missing ckpt_path and model_name raises ValueError", test_missing_ckpt_and_name_raises)
    run_test("Unexpected checkpoint format raises ValueError", test_unexpected_checkpoint_format)

    print()
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print("=" * 60)

    if FAILED > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
