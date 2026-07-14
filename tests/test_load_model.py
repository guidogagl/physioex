"""Tests for physioex.train.models.load -- model loading infrastructure.

Covers dual-format checkpoint support (Lightning .ckpt vs plain .pt),
registry lookup, and error handling.
"""
import pytest
import torch
import torch.nn as nn

from physioex.train.models.load import _get_registry, load_model


def _save(obj, path):
    torch.save(obj, str(path))
    return str(path)


def test_registry_readable():
    df = _get_registry()
    assert len(df) > 0, "Registry is empty"
    required = {"name", "sequence_length", "in_channels", "checkpoint"}
    missing = required - set(df.columns)
    assert not missing, f"Registry missing columns: {missing}"


def test_load_pt_format(tmp_path):
    model = nn.Linear(4, 2)
    w, b = model.weight.data.clone(), model.bias.data.clone()
    path = _save({"model_state_dict": model.state_dict(), "epoch": 0}, tmp_path / "m.pt")

    loaded = load_model(nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path)
    assert torch.allclose(loaded.weight.data, w)
    assert torch.allclose(loaded.bias.data, b)
    assert not loaded.training, "Model should be in eval mode"


def test_load_lightning_format(tmp_path):
    model = nn.Linear(4, 2)
    w, b = model.weight.data.clone(), model.bias.data.clone()
    # Lightning's SleepModule saves weights under an "nn." prefix.
    state = {"nn.weight": w.clone(), "nn.bias": b.clone()}
    path = _save({"state_dict": state}, tmp_path / "m.ckpt")

    loaded = load_model(nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path)
    assert torch.allclose(loaded.weight.data, w)
    assert torch.allclose(loaded.bias.data, b)
    assert not loaded.training


def test_load_raw_state_dict(tmp_path):
    model = nn.Linear(4, 2)
    w, b = model.weight.data.clone(), model.bias.data.clone()
    path = _save(model.state_dict(), tmp_path / "m.pt")

    loaded = load_model(nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path)
    assert torch.allclose(loaded.weight.data, w)
    assert torch.allclose(loaded.bias.data, b)
    assert not loaded.training


def test_registry_lookup_nonexistent_name():
    with pytest.raises(ValueError, match="nonexistent_model_xyz"):
        load_model(
            nn.Linear,
            {"in_features": 4, "out_features": 2, "sequence_length": 21, "in_channels": 1},
            model_name="nonexistent_model_xyz",
        )


def test_registry_lookup_known_name_resolves():
    """A known model name resolves to exactly one registry entry.

    (Rewritten: the old test patched the defunct ``test.train.models.load``
    module path; the resolution logic lives in ``physioex.train.models.load``.)
    """
    table = _get_registry()
    mask = (
        (table["name"] == "seqsleepnet")
        & (table["sequence_length"] == 21)
        & (table["in_channels"] == 1)
    )
    filtered = table[mask]
    assert len(filtered) == 1, f"Expected 1 entry for seqsleepnet/21/1, got {len(filtered)}"
    assert "seqsleepnet" in filtered.iloc[0]["checkpoint"]


def test_model_class_as_string(tmp_path):
    model = nn.Linear(4, 2)
    w = model.weight.data.clone()
    path = _save(model.state_dict(), tmp_path / "m.pt")

    loaded = load_model("torch.nn:Linear", {"in_features": 4, "out_features": 2}, ckpt_path=path)
    assert torch.allclose(loaded.weight.data, w)
    assert not loaded.training


def test_missing_ckpt_and_name_raises():
    with pytest.raises(ValueError, match="ckpt_path|model_name"):
        load_model(nn.Linear, {"in_features": 4, "out_features": 2})


def test_unexpected_checkpoint_format(tmp_path):
    path = _save("not_a_dict", tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Unexpected checkpoint format"):
        load_model(nn.Linear, {"in_features": 4, "out_features": 2}, ckpt_path=path)
