"""
Unit tests for ChannelCache (test/data/cache.py) and collate helpers (test/data/collate.py).

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_cache.py
"""

import os
import sys
import tempfile
import threading
import glob as glob_mod

import numpy as np
import torch

from physioex.data.cache import (
    ChannelCache, SCHEMA_VERSION, _sanitize_name, _encode_physical,
    recommended_dtype, cast_to_cache_dtype, DTYPE_MAP,
)
from physioex.data.collate import dict_collate_fn, stack_channels, is_dict_batch

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
# 1. Cache path construction: basic signal_path
# ---------------------------------------------------------------------------
def test_signal_path_basic():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            p = c.signal_path("hmc", "001", "C4-M2", "abc123")
            expected_parts = [td, "v1", "hmc", "signals", "001", "C4-M2", "abc123", "signal.npy"]
            for part in expected_parts:
                assert part in str(p), f"Expected {part!r} in {str(p)}"
            report("1. signal_path basic structure", True)
    except Exception as exc:
        report("1. signal_path basic structure", False, str(exc))


# ---------------------------------------------------------------------------
# 2. Dataset name with "/" flattened to "__"
# ---------------------------------------------------------------------------
def test_signal_path_slash_dataset():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            p = c.signal_path("AD/AD", "s1", "EEG", "hash")
            assert "AD__AD" in str(p), f"Expected 'AD__AD' in {str(p)}"
            assert "AD/AD" not in str(p), f"Raw slash should not appear in path: {str(p)}"
            report("2. slash in dataset name flattened", True)
    except Exception as exc:
        report("2. slash in dataset name flattened", False, str(exc))


# ---------------------------------------------------------------------------
# 3. Differential physical channel encoded as "C4__M1"
# ---------------------------------------------------------------------------
def test_signal_path_differential():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            p = c.signal_path("hmc", "001", ("C4", "M1"), "h")
            assert "C4__M1" in str(p), f"Expected 'C4__M1' in {str(p)}"
            report("3. differential channel encoding", True)
    except Exception as exc:
        report("3. differential channel encoding", False, str(exc))


# ---------------------------------------------------------------------------
# 4. PHYSIOEX_CACHE_DIR env var override
# ---------------------------------------------------------------------------
def test_env_var_override():
    try:
        with tempfile.TemporaryDirectory() as td:
            custom_root = os.path.join(td, "custom_cache")
            old = os.environ.get("PHYSIOEX_CACHE_DIR")
            try:
                os.environ["PHYSIOEX_CACHE_DIR"] = custom_root
                c = ChannelCache()  # no explicit cache_root
                assert str(c.cache_root) == custom_root, (
                    f"Expected cache_root={custom_root!r}, got {str(c.cache_root)!r}"
                )
            finally:
                if old is None:
                    os.environ.pop("PHYSIOEX_CACHE_DIR", None)
                else:
                    os.environ["PHYSIOEX_CACHE_DIR"] = old
            report("4. PHYSIOEX_CACHE_DIR override", True)
    except Exception as exc:
        report("4. PHYSIOEX_CACHE_DIR override", False, str(exc))


# ---------------------------------------------------------------------------
# 5. Atomic save + load_memmap round trip (float32)
# ---------------------------------------------------------------------------
def test_atomic_save_and_load():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            sig_path = c.signal_path("hmc", "001", "EEG", "pipe1")
            data = np.random.randn(100, 3000).astype(np.float32)
            meta = {"fs_out": 100, "pipeline_spec": "raw"}

            c.atomic_save_array(sig_path, data, meta)

            assert sig_path.exists(), f"Signal file not found: {sig_path}"
            meta_path = sig_path.parent / (sig_path.stem + ".meta.json")
            assert meta_path.exists(), f"Meta file not found: {meta_path}"

            loaded, loaded_meta = c.load_memmap(sig_path)
            assert np.allclose(loaded, data), "Data mismatch after load"
            assert loaded_meta["fs_out"] == 100, f"Meta mismatch: {loaded_meta}"
            assert loaded_meta["dtype"] == "float32"
            assert loaded_meta["shape"] == [100, 3000]
            report("5. atomic_save_array + load_memmap round trip", True)
    except Exception as exc:
        report("5. atomic_save_array + load_memmap round trip", False, str(exc))


# ---------------------------------------------------------------------------
# 6. Concurrent write simulation
# ---------------------------------------------------------------------------
def test_concurrent_write():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            sig_path = c.signal_path("hmc", "001", "EEG", "concurrent")
            errors = []

            def writer(value):
                try:
                    data = np.full((50, 100), value, dtype=np.float32)
                    c.atomic_save_array(sig_path, data, {"writer": value})
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=writer, args=(1.0,))
            t2 = threading.Thread(target=writer, args=(2.0,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert not errors, f"Errors during concurrent write: {errors}"
            assert sig_path.exists(), "File should exist after concurrent writes"
            loaded, meta = c.load_memmap(sig_path)
            # The final file should be a valid array with all same values (from one writer)
            assert loaded.shape == (50, 100), f"Shape mismatch: {loaded.shape}"
            unique = np.unique(loaded)
            assert len(unique) == 1, f"Expected uniform values, got {unique}"
            assert unique[0] in (1.0, 2.0), f"Unexpected value: {unique[0]}"
            report("6. concurrent write simulation", True)
    except Exception as exc:
        report("6. concurrent write simulation", False, str(exc))


# ---------------------------------------------------------------------------
# 7. Save failure cleanup: no orphan tmp files
# ---------------------------------------------------------------------------
def test_save_failure_cleanup():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            sig_path = c.signal_path("hmc", "001", "EEG", "failtest")

            # Ensure the parent directory exists so we can check for orphans
            sig_path.parent.mkdir(parents=True, exist_ok=True)

            # Monkey-patch np.save to raise after the tmp file is created
            original_save = np.save
            def failing_save(*args, **kwargs):
                # Actually write the file first (so a tmp file exists)
                original_save(*args, **kwargs)
                raise IOError("Simulated save failure")

            import physioex.data.cache as cache_mod
            old_np_save = np.save
            np.save = failing_save
            try:
                try:
                    c.atomic_save_array(sig_path, np.zeros((10,), dtype=np.float32), {})
                except IOError:
                    pass  # expected
            finally:
                np.save = old_np_save

            # Check no .tmp. files remain
            tmp_files = glob_mod.glob(str(sig_path.parent / "*.tmp.*"))
            assert len(tmp_files) == 0, f"Orphan tmp files remain: {tmp_files}"
            report("7. save failure cleanup (no orphan tmp files)", True)
    except Exception as exc:
        report("7. save failure cleanup (no orphan tmp files)", False, str(exc))


# ---------------------------------------------------------------------------
# 8. Dtype preservation: bfloat16 round trip (skip if unavailable)
# ---------------------------------------------------------------------------
def test_dtype_bfloat16():
    try:
        if "bfloat16" not in DTYPE_MAP:
            report("8. bfloat16 round trip", True, "SKIPPED (ml_dtypes not available)")
            return

        from ml_dtypes import bfloat16
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            sig_path = c.signal_path("hmc", "001", "EEG", "bf16test")
            data = np.array([1.0, 2.5, -3.0, 0.0], dtype=bfloat16)
            c.atomic_save_array(sig_path, data, {"dtype_test": True})

            loaded, meta = c.load_memmap(sig_path)
            assert meta["dtype"] == "bfloat16", f"Dtype in meta: {meta['dtype']}"
            # bfloat16 arrays need comparison via float32 cast (numpy can't compare custom dtypes directly)
            assert loaded.shape == data.shape, f"Shape mismatch: {loaded.shape} vs {data.shape}"
            assert np.array_equal(
                loaded.astype(np.float32), data.astype(np.float32)
            ), "Data mismatch after bfloat16 round trip"
            report("8. bfloat16 round trip", True)
    except Exception as exc:
        report("8. bfloat16 round trip", False, str(exc))


# ---------------------------------------------------------------------------
# 9. Dtype preservation: float32 round trip
# ---------------------------------------------------------------------------
def test_dtype_float32():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            sig_path = c.signal_path("hmc", "001", "EEG", "f32test")
            data = np.array([1.0, 2.5, -3.0, 0.0], dtype=np.float32)
            c.atomic_save_array(sig_path, data, {"dtype_test": True})

            loaded, meta = c.load_memmap(sig_path)
            assert meta["dtype"] == "float32", f"Dtype in meta: {meta['dtype']}"
            np.testing.assert_array_equal(loaded, data)
            report("9. float32 round trip", True)
    except Exception as exc:
        report("9. float32 round trip", False, str(exc))


# ---------------------------------------------------------------------------
# 10. clear(dataset, subject, pipeline_hash) removes only that pipeline
# ---------------------------------------------------------------------------
def test_clear_pipeline_hash():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            data = np.zeros((10,), dtype=np.float32)

            # Create signals for two channels, two pipelines each
            for ch in ["EEG", "EOG"]:
                for ph in ["hash_a", "hash_b"]:
                    p = c.signal_path("ds", "s1", ch, ph)
                    c.atomic_save_array(p, data, {})

            # Clear only hash_a
            c.clear(dataset="ds", subject="s1", pipeline_hash="hash_a")

            # hash_a dirs should be gone, hash_b should remain
            for ch in ["EEG", "EOG"]:
                assert not c.signal_dir("ds", "s1", ch, "hash_a").exists(), (
                    f"hash_a for {ch} should be removed"
                )
                assert c.signal_dir("ds", "s1", ch, "hash_b").exists(), (
                    f"hash_b for {ch} should remain"
                )
            report("10. clear(dataset, subject, pipeline_hash) scoping", True)
    except Exception as exc:
        report("10. clear(dataset, subject, pipeline_hash) scoping", False, str(exc))


# ---------------------------------------------------------------------------
# 11. clear(dataset, subject) removes the subject subtree
# ---------------------------------------------------------------------------
def test_clear_subject():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            data = np.zeros((10,), dtype=np.float32)

            # Create signal and labels for subject s1 and s2
            for subj in ["s1", "s2"]:
                sp = c.signal_path("ds", subj, "EEG", "h1")
                c.atomic_save_array(sp, data, {})
                lp = c.labels_path("ds", subj)
                c.atomic_save_array(lp, data, {})
                c.save_json(c.header_path("ds", subj), {"test": True})

            c.clear(dataset="ds", subject="s1")

            # s1 signals, labels, header gone
            assert not c.signal_dir("ds", "s1", "EEG", "h1").exists()
            assert not c.labels_path("ds", "s1").exists()
            assert not c.header_path("ds", "s1").exists()

            # s2 untouched
            assert c.signal_path("ds", "s2", "EEG", "h1").exists()
            assert c.labels_path("ds", "s2").exists()
            assert c.header_path("ds", "s2").exists()
            report("11. clear(dataset, subject) removes subject subtree", True)
    except Exception as exc:
        report("11. clear(dataset, subject) removes subject subtree", False, str(exc))


# ---------------------------------------------------------------------------
# 12. clear(dataset) removes the whole dataset
# ---------------------------------------------------------------------------
def test_clear_dataset():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            data = np.zeros((10,), dtype=np.float32)

            for ds in ["ds_a", "ds_b"]:
                sp = c.signal_path(ds, "s1", "EEG", "h1")
                c.atomic_save_array(sp, data, {})

            c.clear(dataset="ds_a")

            assert not c.dataset_root("ds_a").exists(), "ds_a should be removed"
            assert c.dataset_root("ds_b").exists(), "ds_b should remain"
            report("12. clear(dataset) removes whole dataset", True)
    except Exception as exc:
        report("12. clear(dataset) removes whole dataset", False, str(exc))


# ---------------------------------------------------------------------------
# 13. save_json + load_json round trip
# ---------------------------------------------------------------------------
def test_json_round_trip():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            hp = c.header_path("hmc", "001")
            obj = {"channels": ["C4-M2", "EOG"], "fs": 256, "nested": {"a": 1}}
            c.save_json(hp, obj)
            loaded = c.load_json(hp)
            assert loaded == obj, f"Mismatch: {loaded} != {obj}"
            report("13. save_json + load_json round trip", True)
    except Exception as exc:
        report("13. save_json + load_json round trip", False, str(exc))


# ---------------------------------------------------------------------------
# 14. load_json on non-existent file returns None
# ---------------------------------------------------------------------------
def test_json_nonexistent():
    try:
        with tempfile.TemporaryDirectory() as td:
            c = ChannelCache(cache_root=td)
            result = c.load_json(os.path.join(td, "does_not_exist.json"))
            assert result is None, f"Expected None, got {result}"
            report("14. load_json non-existent returns None", True)
    except Exception as exc:
        report("14. load_json non-existent returns None", False, str(exc))


# ---------------------------------------------------------------------------
# 15. dict_collate_fn stacks signals correctly
# ---------------------------------------------------------------------------
def test_collate_stacks_signals():
    try:
        batch = []
        for _ in range(3):
            batch.append({
                "signals": {"EEG": torch.randn(21, 3000)},
                "channel_order": ["EEG"],
                "labels": torch.randint(0, 5, (21,)),
            })

        out = dict_collate_fn(batch)
        assert out["signals"]["EEG"].shape == (3, 21, 3000), (
            f"Expected (3, 21, 3000), got {out['signals']['EEG'].shape}"
        )
        assert out["labels"].shape == (3, 21), f"Labels shape: {out['labels'].shape}"
        report("15. dict_collate_fn stacks signals correctly", True)
    except Exception as exc:
        report("15. dict_collate_fn stacks signals correctly", False, str(exc))


# ---------------------------------------------------------------------------
# 16. dict_collate_fn preserves subject and channel_info as list-of-dicts
# ---------------------------------------------------------------------------
def test_collate_preserves_metadata():
    try:
        batch = []
        for i in range(3):
            batch.append({
                "signals": {"EEG": torch.randn(21, 3000)},
                "channel_order": ["EEG"],
                "labels": torch.randint(0, 5, (21,)),
                "subject": {"id": f"s{i}", "dataset": "hmc"},
                "channel_info": {"EEG": {"fs_in": 256}},
            })

        out = dict_collate_fn(batch)
        assert isinstance(out["subject"], list), f"subject should be list, got {type(out['subject'])}"
        assert len(out["subject"]) == 3
        assert out["subject"][0] == {"id": "s0", "dataset": "hmc"}
        assert isinstance(out["channel_info"], list)
        assert len(out["channel_info"]) == 3
        report("16. dict_collate_fn preserves metadata as list-of-dicts", True)
    except Exception as exc:
        report("16. dict_collate_fn preserves metadata as list-of-dicts", False, str(exc))


# ---------------------------------------------------------------------------
# 17. dict_collate_fn raises on channel_order mismatch
# ---------------------------------------------------------------------------
def test_collate_channel_order_mismatch():
    try:
        # Both items have the same signal keys but different channel_order values.
        # This tests the explicit channel_order consistency check.
        batch = [
            {
                "signals": {"EEG": torch.randn(21, 3000), "EOG": torch.randn(21, 3000)},
                "channel_order": ["EEG", "EOG"],
                "labels": torch.randint(0, 5, (21,)),
            },
            {
                "signals": {"EEG": torch.randn(21, 3000), "EOG": torch.randn(21, 3000)},
                "channel_order": ["EOG", "EEG"],  # different order
                "labels": torch.randint(0, 5, (21,)),
            },
        ]

        raised = False
        try:
            dict_collate_fn(batch)
        except ValueError as e:
            raised = True
            assert "mismatch" in str(e).lower(), f"Error message should mention mismatch: {e}"

        assert raised, "Should have raised ValueError for channel_order mismatch"
        report("17. dict_collate_fn raises on channel_order mismatch", True)
    except Exception as exc:
        report("17. dict_collate_fn raises on channel_order mismatch", False, str(exc))


# ---------------------------------------------------------------------------
# 18. stack_channels produces (B, L, C, ...) with correct order
# ---------------------------------------------------------------------------
def test_stack_channels():
    try:
        B, L, T = 4, 21, 3000
        eeg = torch.randn(B, L, T)
        eog = torch.randn(B, L, T)
        emg = torch.randn(B, L, T)

        batch = {
            "signals": {"EEG": eeg, "EOG": eog, "EMG": emg},
            "channel_order": ["EEG", "EOG", "EMG"],
        }

        result = stack_channels(batch)
        assert result.shape == (B, L, 3, T), f"Expected (4, 21, 3, 3000), got {result.shape}"
        # Verify order: dim=2 index 0 should be EEG, 1 EOG, 2 EMG
        assert torch.equal(result[:, :, 0, :], eeg)
        assert torch.equal(result[:, :, 1, :], eog)
        assert torch.equal(result[:, :, 2, :], emg)
        report("18. stack_channels produces (B, L, C, ...) with correct order", True)
    except Exception as exc:
        report("18. stack_channels produces (B, L, C, ...) with correct order", False, str(exc))


# ---------------------------------------------------------------------------
# 19. is_dict_batch returns True/False correctly
# ---------------------------------------------------------------------------
def test_is_dict_batch():
    try:
        dict_batch = {"signals": {"EEG": torch.randn(4, 21, 3000)}, "channel_order": ["EEG"]}
        tuple_batch = (torch.randn(4, 21, 3000), torch.randint(0, 5, (4, 21)))

        assert is_dict_batch(dict_batch) is True, "dict batch should return True"
        assert is_dict_batch(tuple_batch) is False, "tuple batch should return False"
        assert is_dict_batch({"signals": {}}) is False, "missing channel_order should return False"
        assert is_dict_batch(42) is False, "non-dict should return False"
        report("19. is_dict_batch returns True/False correctly", True)
    except Exception as exc:
        report("19. is_dict_batch returns True/False correctly", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running cache + collate tests")
    print("=" * 60)

    test_signal_path_basic()
    test_signal_path_slash_dataset()
    test_signal_path_differential()
    test_env_var_override()
    test_atomic_save_and_load()
    test_concurrent_write()
    test_save_failure_cleanup()
    test_dtype_bfloat16()
    test_dtype_float32()
    test_clear_pipeline_hash()
    test_clear_subject()
    test_clear_dataset()
    test_json_round_trip()
    test_json_nonexistent()
    test_collate_stacks_signals()
    test_collate_preserves_metadata()
    test_collate_channel_order_mismatch()
    test_stack_channels()
    test_is_dict_batch()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
