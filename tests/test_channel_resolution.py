"""
Unit tests for EDF channel resolution logic.

Tests the pure resolution algorithm with mock inputs (no real EDF files).

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_channel_resolution.py
"""

import sys

from physioex.data.readers.edf import (
    resolve_channels,
    ResolvedChannel,
    ChannelNotAvailableError,
    DEFAULT_PREFERENCES,
    KNOWN_MODALITIES,
    _classify_modality,
)

def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# Test 1: Generic modality -- "EEG" resolves via default preference list
# ---------------------------------------------------------------------------
def test_generic_modality_eeg():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0}
        result = resolve_channels(["EEG"], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        assert rc.physical == "C4-M1", f"Expected 'C4-M1', got {rc.physical!r}"
        assert rc.modality == "EEG"
        assert rc.is_differential is False
        report("1: Generic modality EEG resolves to C4-M1", True)
    except Exception as exc:
        report("1: Generic modality EEG resolves to C4-M1", False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: Specific channel -- exact name match
# ---------------------------------------------------------------------------
def test_specific_channel():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0}
        result = resolve_channels(["C4-M1"], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        assert rc.physical == "C4-M1", f"Expected 'C4-M1', got {rc.physical!r}"
        assert rc.is_differential is False
        report("2: Specific channel C4-M1", True)
    except Exception as exc:
        report("2: Specific channel C4-M1", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: Case-insensitive specific match
# ---------------------------------------------------------------------------
def test_case_insensitive_specific():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0}
        result = resolve_channels(["c4-m1"], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        # physical should be the requested string (case as passed), resolved by index
        assert rc.physical == "c4-m1", f"Expected 'c4-m1', got {rc.physical!r}"
        report("3: Case-insensitive specific 'c4-m1' matches 'C4-M1'", True)
    except Exception as exc:
        report("3: Case-insensitive specific 'c4-m1' matches 'C4-M1'", False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: Two generic EEG requests resolve to distinct channels
# ---------------------------------------------------------------------------
def test_two_generic_eeg():
    try:
        available = ["C4-M1", "C3-M2", "EOG"]
        fs_map = {"C4-M1": 256.0, "C3-M2": 256.0, "EOG": 256.0}
        result = resolve_channels(["EEG", "EEG"], available, fs_map)
        assert len(result) == 2
        phys0 = result[0].physical
        phys1 = result[1].physical
        assert phys0 != phys1, f"Both resolved to {phys0!r}"
        assert phys0 == "C4-M1", f"First EEG: expected 'C4-M1', got {phys0!r}"
        assert phys1 == "C3-M2", f"Second EEG: expected 'C3-M2', got {phys1!r}"
        report("4: Two generic EEG resolve to distinct channels", True)
    except Exception as exc:
        report("4: Two generic EEG resolve to distinct channels", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: Differential pair via tuple in preference list
# ---------------------------------------------------------------------------
def test_differential_pair():
    try:
        available = ["C4", "M1", "EOG"]
        fs_map = {"C4": 256.0, "M1": 256.0, "EOG": 256.0}
        result = resolve_channels(["EEG"], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        assert rc.is_differential is True, f"Expected differential, got is_differential={rc.is_differential}"
        assert isinstance(rc.physical, tuple), f"Expected tuple physical, got {type(rc.physical)}"
        assert rc.physical == ("C4", "M1"), f"Expected ('C4', 'M1'), got {rc.physical!r}"
        assert rc.modality == "EEG"
        report("5: Differential pair ('C4', 'M1') resolved", True)
    except Exception as exc:
        report("5: Differential pair ('C4', 'M1') resolved", False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: Custom preference via dict form
# ---------------------------------------------------------------------------
def test_custom_preference_dict():
    try:
        available = ["C3-M2", "C4-M1", "EOG"]
        fs_map = {"C3-M2": 100.0, "C4-M1": 100.0, "EOG": 100.0}
        result = resolve_channels(
            [{"modality": "EEG", "preference": ["C3-M2"]}],
            available, fs_map,
        )
        assert len(result) == 1
        rc = result[0]
        assert rc.physical == "C3-M2", f"Expected 'C3-M2', got {rc.physical!r}"
        assert rc.modality == "EEG"
        report("6: Custom preference dict picks C3-M2", True)
    except Exception as exc:
        report("6: Custom preference dict picks C3-M2", False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: Dict form with 'name' key (specific channel)
# ---------------------------------------------------------------------------
def test_dict_name():
    try:
        available = ["EOG", "C4-M1"]
        fs_map = {"EOG": 100.0, "C4-M1": 100.0}
        result = resolve_channels([{"name": "EOG"}], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        assert rc.physical == "EOG", f"Expected 'EOG', got {rc.physical!r}"
        report("7: Dict {'name': 'EOG'} resolves to EOG", True)
    except Exception as exc:
        report("7: Dict {'name': 'EOG'} resolves to EOG", False, str(exc))


# ---------------------------------------------------------------------------
# Test 8: Unresolvable modality raises ChannelNotAvailableError
# ---------------------------------------------------------------------------
def test_unresolvable_raises():
    try:
        available = ["Respiration"]
        fs_map = {"Respiration": 25.0}
        try:
            resolve_channels(["EEG"], available, fs_map)
            report("8: Unresolvable EEG raises", False, "No exception raised")
        except ChannelNotAvailableError:
            report("8: Unresolvable EEG raises ChannelNotAvailableError", True)
        except Exception as exc:
            report("8: Unresolvable EEG raises", False, f"Wrong exception: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report("8: Unresolvable EEG raises", False, str(exc))


# ---------------------------------------------------------------------------
# Test 9: Second duplicate EEG with no more options raises
# ---------------------------------------------------------------------------
def test_second_eeg_no_options_raises():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0}
        try:
            resolve_channels(["EEG", "EEG"], available, fs_map)
            report("9: Second EEG with no more options raises", False, "No exception raised")
        except ChannelNotAvailableError:
            report("9: Second EEG with no more options raises ChannelNotAvailableError", True)
        except Exception as exc:
            report("9: Second EEG with no more options raises", False, f"Wrong exception: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report("9: Second EEG with no more options raises", False, str(exc))


# ---------------------------------------------------------------------------
# Test 10: Specific double-claim raises
# ---------------------------------------------------------------------------
def test_specific_double_claim_raises():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0}
        try:
            resolve_channels(["C4-M1", "C4-M1"], available, fs_map)
            report("10: Specific double-claim raises", False, "No exception raised")
        except ChannelNotAvailableError:
            report("10: Specific double-claim raises ChannelNotAvailableError", True)
        except Exception as exc:
            report("10: Specific double-claim raises", False, f"Wrong exception: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report("10: Specific double-claim raises", False, str(exc))


# ---------------------------------------------------------------------------
# Test 11: Unknown request type raises TypeError
# ---------------------------------------------------------------------------
def test_unknown_request_type_raises():
    try:
        available = ["C4-M1"]
        fs_map = {"C4-M1": 100.0}
        try:
            resolve_channels([42], available, fs_map)
            report("11: Unknown request type raises", False, "No exception raised")
        except TypeError:
            report("11: Unknown request type (int) raises TypeError", True)
        except Exception as exc:
            report("11: Unknown request type raises", False, f"Wrong exception: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report("11: Unknown request type raises", False, str(exc))


# ---------------------------------------------------------------------------
# Test 12: Dict without 'name' or 'modality' raises ValueError
# ---------------------------------------------------------------------------
def test_dict_without_required_keys_raises():
    try:
        available = ["C4-M1"]
        fs_map = {"C4-M1": 100.0}
        try:
            resolve_channels([{"foo": "bar"}], available, fs_map)
            report("12: Dict without name/modality raises", False, "No exception raised")
        except ValueError:
            report("12: Dict without name/modality raises ValueError", True)
        except Exception as exc:
            report("12: Dict without name/modality raises", False, f"Wrong exception: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report("12: Dict without name/modality raises", False, str(exc))


# ---------------------------------------------------------------------------
# Test 13: fs_in propagation
# ---------------------------------------------------------------------------
def test_fs_in_propagation():
    try:
        available = ["C4-M1", "EOG"]
        fs_map = {"C4-M1": 256.0, "EOG": 128.0}
        result = resolve_channels(["EEG", "EOG"], available, fs_map)
        assert result[0].fs_in == 256.0, f"EEG fs_in: expected 256.0, got {result[0].fs_in}"
        assert result[1].fs_in == 128.0, f"EOG fs_in: expected 128.0, got {result[1].fs_in}"
        report("13: fs_in propagation (256 for EEG, 128 for EOG)", True)
    except Exception as exc:
        report("13: fs_in propagation", False, str(exc))


# ---------------------------------------------------------------------------
# Test 14: Modality classification of specific channel
# ---------------------------------------------------------------------------
def test_modality_classification_of_specific():
    try:
        available = ["C4-M1", "EOG", "EMG"]
        fs_map = {"C4-M1": 100.0, "EOG": 100.0, "EMG": 100.0}
        # "C4-M1" appears in DEFAULT_EEG_PREFERENCES, so should be classified as EEG
        result = resolve_channels(["C4-M1"], available, fs_map)
        assert len(result) == 1
        rc = result[0]
        assert rc.modality == "EEG", f"Expected modality 'EEG', got {rc.modality!r}"
        # "EOG" appears in DEFAULT_EOG_PREFERENCES
        result2 = resolve_channels(["EOG"], available, fs_map)
        rc2 = result2[0]
        assert rc2.modality == "EOG", f"Expected modality 'EOG', got {rc2.modality!r}"
        report("14: Modality classification of specific channels", True)
    except Exception as exc:
        report("14: Modality classification of specific channels", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
