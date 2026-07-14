"""Unit tests for pure helpers in physioex.models.pretrained / embed.

The heavy paths (load_from_pretrained, extract_embeddings, linear_probe) need
checkpoints / HuggingFace and are covered by gated tests elsewhere; here we
pin the small pure functions that underpin them.
"""
from pathlib import Path

import numpy as np
import pytest

from physioex.models.pretrained import _resolve_class
from physioex.models.embed import _hf_repo_id, _load_npy_as_float32


# ── pretrained._resolve_class ────────────────────────────────────────

def test_resolve_class_valid_spec():
    cls = _resolve_class("physioex.models.chambon2018:Chambon2018Net")
    from physioex.models.chambon2018 import Chambon2018Net
    assert cls is Chambon2018Net


def test_resolve_class_bad_class_raises():
    with pytest.raises(AttributeError):
        _resolve_class("physioex.models.chambon2018:NoSuchClass")


def test_resolve_class_bad_module_raises():
    with pytest.raises(ModuleNotFoundError):
        _resolve_class("physioex.models.does_not_exist:Foo")


# ── embed._hf_repo_id ────────────────────────────────────────────────

def test_hf_repo_id_format():
    repo = _hf_repo_id("cbramod")
    assert repo.endswith("/cbramod-embeddings")
    assert "/" in repo


# ── embed._load_npy_as_float32 ───────────────────────────────────────

def test_load_npy_float_cast(tmp_path):
    p = tmp_path / "f.npy"
    np.save(p, np.array([[1.0, 2.0]], dtype=np.float64))
    out = _load_npy_as_float32(p)
    assert out.dtype == np.float32
    assert np.allclose(out, [[1.0, 2.0]])


def test_load_npy_int_preserved(tmp_path):
    p = tmp_path / "i.npy"
    np.save(p, np.array([1, 2, 3], dtype=np.int16))
    out = _load_npy_as_float32(p)
    assert out.dtype == np.int16  # int16 labels preserved, not cast
    assert out.tolist() == [1, 2, 3]
