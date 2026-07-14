"""Unit tests for physioex.data.modality (channel-to-modality inference).

Covers:
  - ModalityType enum: values, ``__str__``, count invariants
  - MODALITY_TYPES / N_MODALITY_TYPES legacy mirrors
  - infer_channel_modality: hint precedence + name-based inference across
    every modality bucket, plus the OTHER/DEVICE fallbacks.
"""
import pytest

from physioex.data.modality import (
    ModalityType,
    MODALITY_TYPES,
    N_MODALITY_TYPES,
    infer_channel_modality,
)


# ---------------------------------------------------------------------------
# 1. ModalityType enum + legacy mirrors
# ---------------------------------------------------------------------------

def test_modality_type_is_intenum():
    assert ModalityType.EEG == 0
    assert int(ModalityType.OTHER) == 14
    assert str(ModalityType.EEG) == "EEG"
    assert str(ModalityType.SPO2) == "SPO2"


def test_modality_count_invariants():
    # 15 types total (EEG..OTHER); docstring history mentions the TEMP addition.
    assert N_MODALITY_TYPES == len(ModalityType) == 15
    assert MODALITY_TYPES == {m.name: int(m) for m in ModalityType}
    # Legacy dict maps name -> integer value consistently.
    assert MODALITY_TYPES["EEG"] == 0
    assert MODALITY_TYPES["OTHER"] == 14


# ---------------------------------------------------------------------------
# 2. Hint precedence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hint", ["EEG", "EOG", "EMG", "ECG"])
def test_hint_takes_precedence(hint):
    # Even an EMG-looking name is overridden by a valid hint.
    assert infer_channel_modality("Chin1", hint=hint) == ModalityType[hint]


def test_invalid_hint_falls_back_to_name():
    # A hint that is not in MODALITY_TYPES is ignored; name inference wins.
    assert infer_channel_modality("C4-M1", hint="NOTAHINT") == ModalityType.EEG


# ---------------------------------------------------------------------------
# 3. Name-based inference across modalities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        # EEG
        ("EEG C4-M1", ModalityType.EEG),
        ("C4-M2", ModalityType.EEG),
        ("Fp1", ModalityType.EEG),
        ("O2", ModalityType.EEG),
        # EOG
        ("EOG-L", ModalityType.EOG),
        ("LOC", ModalityType.EOG),
        ("ROC", ModalityType.EOG),
        # EMG
        ("EMG Chin", ModalityType.EMG),
        ("Chin1", ModalityType.EMG),
        # ECG
        ("ECG", ModalityType.ECG),
        ("EKG", ModalityType.ECG),
        # LEG
        ("Leg/L", ModalityType.LEG),
        ("Tibial", ModalityType.LEG),
        # RESP
        ("Nasal Flow", ModalityType.RESP),
        ("Thor", ModalityType.RESP),
        ("Abdo", ModalityType.RESP),
        # SPO2
        ("SpO2", ModalityType.SPO2),
        ("Pleth", ModalityType.SPO2),
        # HR
        ("HR", ModalityType.HR),
        ("Pulse", ModalityType.HR),
        # POS
        ("Position", ModalityType.POS),
        ("Body", ModalityType.POS),
        # LIGHT / SOUND
        ("Light", ModalityType.LIGHT),
        # "Snore" alone hits the RESP "SNOR" keyword; a bare Mic is SOUND.
        ("Mic", ModalityType.SOUND),
        # DEVICE
        ("Marker", ModalityType.DEVICE),
        ("Battery", ModalityType.DEVICE),
        # OTHER (unrecognized)
        ("SomethingWeird123XYZ", ModalityType.OTHER),
    ],
)
def test_infer_by_name(name, expected):
    assert infer_channel_modality(name) == expected


def test_inference_is_case_and_separator_insensitive():
    # Separators [-_\s.()#/] are stripped and casing normalized.
    assert infer_channel_modality("eeg c4-m1") == ModalityType.EEG
    assert infer_channel_modality("E E G") == ModalityType.EEG
    assert infer_channel_modality("SpO2") == infer_channel_modality("spo2")


def test_empty_channel_name_is_device():
    # Empty/whitespace collapses to "" which the DEVICE bucket claims.
    assert infer_channel_modality("") == ModalityType.DEVICE
    assert infer_channel_modality("   ") == ModalityType.DEVICE
