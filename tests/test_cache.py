"""Tests for ChannelCache (physioex/data/cache.py) and the collate helpers
(physioex/data/collate.py)."""
import glob as glob_mod

import numpy as np
import torch

from physioex.data.cache import ChannelCache, DTYPE_MAP
from physioex.data.collate import dict_collate_fn, is_dict_batch, stack_channels


# ---------------------------------------------------------------------------
# Cache path construction
# ---------------------------------------------------------------------------

def test_signal_path_basic(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    p = str(c.signal_path("hmc", "001", "C4-M2", "abc123"))
    for part in ["v1", "hmc", "signals", "001", "C4-M2", "abc123", "signal.npy"]:
        assert part in p, f"Expected {part!r} in {p}"


def test_signal_path_slash_dataset(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    p = str(c.signal_path("AD/AD", "s1", "EEG", "hash"))
    assert "AD__AD" in p and "AD/AD" not in p


def test_signal_path_differential(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    p = str(c.signal_path("hmc", "001", ("C4", "M1"), "h"))
    assert "C4__M1" in p


def test_env_var_override(tmp_path, monkeypatch):
    custom_root = str(tmp_path / "custom_cache")
    monkeypatch.setenv("PHYSIOEX_CACHE_DIR", custom_root)
    c = ChannelCache()  # no explicit cache_root
    assert str(c.cache_root) == custom_root


# ---------------------------------------------------------------------------
# Save / load round trips
# ---------------------------------------------------------------------------

def test_atomic_save_and_load(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    sig_path = c.signal_path("hmc", "001", "EEG", "pipe1")
    data = np.random.randn(100, 3000).astype(np.float32)
    c.atomic_save_array(sig_path, data, {"fs_out": 100, "pipeline_spec": "raw"})

    assert sig_path.exists()
    assert (sig_path.parent / (sig_path.stem + ".meta.json")).exists()

    loaded, meta = c.load_memmap(sig_path)
    assert np.allclose(loaded, data)
    assert meta["fs_out"] == 100
    assert meta["dtype"] == "float32"
    assert meta["shape"] == [100, 3000]


def test_concurrent_write(tmp_path):
    import threading

    c = ChannelCache(cache_root=str(tmp_path))
    sig_path = c.signal_path("hmc", "001", "EEG", "concurrent")
    errors = []

    def writer(value):
        try:
            c.atomic_save_array(sig_path, np.full((50, 100), value, dtype=np.float32), {"writer": value})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(v,)) for v in (1.0, 2.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent write: {errors}"
    loaded, _ = c.load_memmap(sig_path)
    assert loaded.shape == (50, 100)
    unique = np.unique(loaded)
    assert len(unique) == 1 and unique[0] in (1.0, 2.0)


def test_save_failure_cleanup(tmp_path, monkeypatch):
    c = ChannelCache(cache_root=str(tmp_path))
    sig_path = c.signal_path("hmc", "001", "EEG", "failtest")
    sig_path.parent.mkdir(parents=True, exist_ok=True)

    original_save = np.save

    def failing_save(*args, **kwargs):
        original_save(*args, **kwargs)  # create the tmp file first
        raise IOError("Simulated save failure")

    monkeypatch.setattr(np, "save", failing_save)
    try:
        c.atomic_save_array(sig_path, np.zeros((10,), dtype=np.float32), {})
    except IOError:
        pass  # expected

    assert not glob_mod.glob(str(sig_path.parent / "*.tmp.*")), "orphan tmp files remain"


def test_dtype_bfloat16(tmp_path):
    if "bfloat16" not in DTYPE_MAP:
        import pytest

        pytest.skip("ml_dtypes not available")
    from ml_dtypes import bfloat16

    c = ChannelCache(cache_root=str(tmp_path))
    sig_path = c.signal_path("hmc", "001", "EEG", "bf16test")
    data = np.array([1.0, 2.5, -3.0, 0.0], dtype=bfloat16)
    c.atomic_save_array(sig_path, data, {"dtype_test": True})

    loaded, meta = c.load_memmap(sig_path)
    assert meta["dtype"] == "bfloat16"
    assert loaded.shape == data.shape
    assert np.array_equal(loaded.astype(np.float32), data.astype(np.float32))


def test_dtype_float32(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    sig_path = c.signal_path("hmc", "001", "EEG", "f32test")
    data = np.array([1.0, 2.5, -3.0, 0.0], dtype=np.float32)
    c.atomic_save_array(sig_path, data, {"dtype_test": True})

    loaded, meta = c.load_memmap(sig_path)
    assert meta["dtype"] == "float32"
    np.testing.assert_array_equal(loaded, data)


# ---------------------------------------------------------------------------
# clear() scoping
# ---------------------------------------------------------------------------

def test_clear_pipeline_hash(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    data = np.zeros((10,), dtype=np.float32)
    for ch in ["EEG", "EOG"]:
        for ph in ["hash_a", "hash_b"]:
            c.atomic_save_array(c.signal_path("ds", "s1", ch, ph), data, {})

    c.clear(dataset="ds", subject="s1", pipeline_hash="hash_a")

    for ch in ["EEG", "EOG"]:
        assert not c.signal_dir("ds", "s1", ch, "hash_a").exists()
        assert c.signal_dir("ds", "s1", ch, "hash_b").exists()


def test_clear_subject(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    data = np.zeros((10,), dtype=np.float32)
    for subj in ["s1", "s2"]:
        c.atomic_save_array(c.signal_path("ds", subj, "EEG", "h1"), data, {})
        c.atomic_save_array(c.labels_path("ds", subj), data, {})
        c.save_json(c.header_path("ds", subj), {"test": True})

    c.clear(dataset="ds", subject="s1")

    assert not c.signal_dir("ds", "s1", "EEG", "h1").exists()
    assert not c.labels_path("ds", "s1").exists()
    assert not c.header_path("ds", "s1").exists()
    assert c.signal_path("ds", "s2", "EEG", "h1").exists()
    assert c.labels_path("ds", "s2").exists()
    assert c.header_path("ds", "s2").exists()


def test_clear_dataset(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    data = np.zeros((10,), dtype=np.float32)
    for ds in ["ds_a", "ds_b"]:
        c.atomic_save_array(c.signal_path(ds, "s1", "EEG", "h1"), data, {})

    c.clear(dataset="ds_a")
    assert not c.dataset_root("ds_a").exists()
    assert c.dataset_root("ds_b").exists()


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------

def test_json_round_trip(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    hp = c.header_path("hmc", "001")
    obj = {"channels": ["C4-M2", "EOG"], "fs": 256, "nested": {"a": 1}}
    c.save_json(hp, obj)
    assert c.load_json(hp) == obj


def test_json_nonexistent(tmp_path):
    c = ChannelCache(cache_root=str(tmp_path))
    assert c.load_json(str(tmp_path / "does_not_exist.json")) is None


# ---------------------------------------------------------------------------
# collate helpers
# ---------------------------------------------------------------------------

def test_collate_stacks_signals():
    batch = [
        {
            "signals": {"EEG": torch.randn(21, 3000)},
            "channel_order": ["EEG"],
            "labels": torch.randint(0, 5, (21,)),
        }
        for _ in range(3)
    ]
    out = dict_collate_fn(batch)
    assert out["signals"]["EEG"].shape == (3, 21, 3000)
    assert out["labels"].shape == (3, 21)


def test_collate_preserves_metadata():
    batch = [
        {
            "signals": {"EEG": torch.randn(21, 3000)},
            "channel_order": ["EEG"],
            "labels": torch.randint(0, 5, (21,)),
            "subject": {"id": f"s{i}", "dataset": "hmc"},
            "channel_info": {"EEG": {"fs_in": 256}},
        }
        for i in range(3)
    ]
    out = dict_collate_fn(batch)
    assert isinstance(out["subject"], list) and len(out["subject"]) == 3
    assert out["subject"][0] == {"id": "s0", "dataset": "hmc"}
    assert isinstance(out["channel_info"], list) and len(out["channel_info"]) == 3


def test_collate_channel_order_is_deterministic():
    """Rewritten (was stale): dict_collate_fn does NOT validate/raise on a
    per-item channel_order mismatch; it derives a deterministic canonical order
    from the sorted union of signal keys. Two items with the same keys in
    different order collate cleanly to that canonical order."""
    batch = [
        {
            "signals": {"EEG": torch.randn(21, 3000), "EOG": torch.randn(21, 3000)},
            "channel_order": ["EEG", "EOG"],
            "labels": torch.randint(0, 5, (21,)),
        },
        {
            "signals": {"EEG": torch.randn(21, 3000), "EOG": torch.randn(21, 3000)},
            "channel_order": ["EOG", "EEG"],  # different input order
            "labels": torch.randint(0, 5, (21,)),
        },
    ]
    out = dict_collate_fn(batch)
    # No exception; canonical order contains both channels deterministically.
    assert set(out["channel_order"]) == {"EEG", "EOG"}
    assert dict_collate_fn(batch)["channel_order"] == out["channel_order"]  # stable
    assert out["signals"]["EEG"].shape == (2, 21, 3000)
    assert out["signals"]["EOG"].shape == (2, 21, 3000)


def test_stack_channels():
    B, L, T = 4, 21, 3000
    eeg, eog, emg = torch.randn(B, L, T), torch.randn(B, L, T), torch.randn(B, L, T)
    batch = {"signals": {"EEG": eeg, "EOG": eog, "EMG": emg}, "channel_order": ["EEG", "EOG", "EMG"]}
    result = stack_channels(batch)
    assert result.shape == (B, L, 3, T)
    assert torch.equal(result[:, :, 0, :], eeg)
    assert torch.equal(result[:, :, 1, :], eog)
    assert torch.equal(result[:, :, 2, :], emg)


def test_is_dict_batch():
    assert is_dict_batch({"signals": {"EEG": torch.randn(4, 21, 3000)}, "channel_order": ["EEG"]}) is True
    assert is_dict_batch((torch.randn(4, 21, 3000), torch.randint(0, 5, (4, 21)))) is False
    assert is_dict_batch({"signals": {}}) is False
    assert is_dict_batch(42) is False
