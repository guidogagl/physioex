"""Clean-channel recipes (CLEAN_MONTAGE) resolved against fake headers with the labels of each dataset."""

import numpy as np
import pytest

from physioex.data.cleaning import (
    CleanMontage,
    Fallback,
    read_derivation,
    resolve_montage,
)
from physioex.data.datasets.alzheimers import AlzheimersDataset
from physioex.data.datasets.ccshs import CCSHSDataset
from physioex.data.datasets.dcsm import DCSMDataset
from physioex.data.datasets.hmc import HMCDataset
from physioex.data.datasets.hpap import HPAPDataset
from physioex.data.datasets.mass import MASSDataset
from physioex.data.datasets.mesa import MESADataset
from physioex.data.datasets.mros import MrOSDataset
from physioex.data.datasets.parkinsons import ParkinsonsDataset
from physioex.data.datasets.shhs import SHHSDataset
from physioex.data.datasets.sleepedf import SleepEDFDataset
from physioex.data.datasets.wsc import WSCDataset

_ALZ = [
    f"EEG {e}-REF" for e in "C3 C4 F3 F4 O1 O2 Fp1 Fp2 Fz Pz T9 T10 EOG1 EOG2".split()
]
_MASS = [f"EEG {e}-CLE" for e in "C3 C4 F3 F4 O1 O2 Fz Cz Pz Fp1 Fp2".split()]

# (dataset class, typical labels of one recording, expected EEG, expected EOG, expected level)
CASES = [
    (
        SHHSDataset,
        ["EEG", "EEG(sec)", "EOG(L)", "EOG(R)", "EMG", "ECG"],
        "EEG",
        "EOG(L) - EOG(R)",
        "poor",
    ),
    (
        MrOSDataset,
        [
            "C3",
            "C4",
            "A1",
            "A2",
            "LOC",
            "ROC",
            "L Chin",
            "R Chin",
            "ECG L",
            "ECG R",
            "Leg L",
            "Leg R",
        ],
        "C4 - A1",
        "LOC - ROC",
        "poor",
    ),
    (
        MESADataset,
        ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EMG", "EKG", "Leg"],
        "EEG3",
        "EOG-L - EOG-R",
        "poor",
    ),
    (
        WSCDataset,
        ["C3_M2", "F3_M2", "O1_M2", "E1", "E2", "chin", "lleg_r"],
        "C3_M2",
        "E1 - E2",
        "poor",
    ),
    (
        CCSHSDataset,
        ["C3", "C4", "A1", "A2", "LOC", "ROC", "EMG1", "EMG2", "EMG3", "ECG1", "ECG2"],
        "C4 - A1",
        "LOC - ROC",
        "poor",
    ),
    (
        DCSMDataset,
        [
            "C3-M2",
            "C4-M1",
            "F3-M2",
            "F4-M1",
            "O1-M2",
            "O2-M1",
            "E1-M2",
            "E2-M2",
            "CHIN",
            "LAT",
            "RAT",
            "ECG-II",
        ],
        "C4-M1",
        "E1-M2 - E2-M2",
        "rich",
    ),
    (
        SleepEDFDataset,
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
        "EEG Fpz-Cz",
        "EOG horizontal",
        "envelope",
    ),
    (
        HPAPDataset,
        [
            "C3",
            "C4",
            "F3",
            "F4",
            "O1",
            "O2",
            "M1",
            "M2",
            "E1",
            "E2",
            "Lchin",
            "Rchin",
            "Cchin",
            "ECG1",
            "ECG3",
        ],
        "C4 - M1",
        "E1 - E2",
        "rich",
    ),
    (
        HMCDataset,
        [
            "EEG C3-M2",
            "EEG C4-M1",
            "EEG F4-M1",
            "EEG O2-M1",
            "EOG E1-M2",
            "EOG E2-M2",
            "EMG chin",
            "ECG",
        ],
        "EEG C4-M1",
        "EOG E1-M2 - EOG E2-M2",
        "rich",
    ),
    (
        MASSDataset,
        _MASS
        + ["EOG Left Horiz", "EOG Right Horiz", "EMG Chin1", "EMG Chin2", "ECG I"],
        "EEG C4-CLE",
        "EOG Left Horiz - EOG Right Horiz",
        "rich",
    ),
    (
        AlzheimersDataset,
        _ALZ + ["EMG Chin", "ECG V1"],
        "EEG C4-REF - EEG T9-REF",
        "EEG EOG1-REF - EEG EOG2-REF",
        "rich",
    ),
    (
        ParkinsonsDataset,
        [f"EEG {e}" for e in "C3-A2 C4-A1 F3-A2 F4-A1 O1-A2 O2-A1 T5-A1 T6-A1".split()]
        + ["EOG Left", "EOG right", "EMG Chin", "ECG V1"],
        "EEG C4-A1",
        "EOG Left - EOG right",
        "rich",
    ),
]


@pytest.mark.parametrize(
    "cls,labels,eeg,eog,level", CASES, ids=[c[0].__name__ for c in CASES]
)
def test_recipe_resolves(cls, labels, eeg, eog, level):
    m = cls.CLEAN_MONTAGE
    assert isinstance(m, CleanMontage)
    r = resolve_montage(m, labels, {lab: 256.0 for lab in labels})
    assert r.derivations["eeg"].label == eeg
    assert r.derivations["eog"].label == eog
    assert r.level == level, r.reasons


def test_every_recipe_has_a_stable_distinct_spec():
    specs = [c[0].CLEAN_MONTAGE.spec() for c in CASES]
    assert specs == [c[0].CLEAN_MONTAGE.spec() for c in CASES]
    assert len(set(specs)) == len(specs)


def test_hpap_opposite_mastoids_rebuild_e1_minus_e2():
    labels = ["C4-M1", "C3-M2", "E1-M2", "E2-M1", "M1-M2"]
    r = resolve_montage(
        HPAPDataset.CLEAN_MONTAGE, labels, {lab: 200.0 for lab in labels}
    )
    assert r.derivations["eog"].label == "(E1-M2) - (E2-M1) - (M1-M2)"
    assert r.derivations["eog"].level == "rich"
    # without M1-M2 only the opposite-mastoid difference is left, and the night drops to poor
    r = resolve_montage(
        HPAPDataset.CLEAN_MONTAGE, labels[:4], {lab: 200.0 for lab in labels}
    )
    assert r.derivations["eog"].label == "E1-M2 - E2-M1"
    assert r.level == "poor"


def test_hpap_alternative_eye_labels():
    labels = ["C3", "C4", "M1", "M2", "E-1", "E-2"]
    r = resolve_montage(
        HPAPDataset.CLEAN_MONTAGE, labels, {lab: 256.0 for lab in labels}
    )
    assert r.derivations["eog"].label == "E-1 - E-2"


def test_missing_eog_and_mixed_rates():
    m = CleanMontage(eeg=[("C4", "M1")], eog=[("E1", "E2")], level="poor")
    r = resolve_montage(m, ["C4", "M1", "E1"], {"C4": 256, "M1": 256, "E1": 256})
    assert r.derivations["eog"] is None and r.level == "missing"
    # terms at different sampling rates are not combined
    r = resolve_montage(
        m, ["C4", "M1", "E1", "E2"], {"C4": 256, "M1": 128, "E1": 256, "E2": 256}
    )
    assert r.derivations["eeg"] is None


def test_extra_identical_to_canonical_is_dropped():
    m = CleanMontage(
        eeg=[("C4", "A1"), ("C3", "A2")],
        eog=["E"],
        extra={"C3": [("C3", "A2")]},
        extra_eeg=["C3"],
    )
    r = resolve_montage(m, ["C3", "A2", "E"], {"C3": 1, "A2": 1, "E": 1})
    assert r.derivations["eeg"].label == "C3 - A2" and "C3" not in r.extra


class _FakeReader:
    def __init__(self, sig, units):
        self.sig, self.units = sig, units

    def readSignal(self, i):
        return self.sig[i]

    def getPhysicalDimension(self, i):
        return self.units[i]


def test_read_derivation_is_exact_and_scales_units():
    rng = np.random.default_rng(0)
    e1, e2, m1, m2 = (rng.standard_normal(1000) for _ in range(4))
    labels = ["E1-M2", "E2-M1", "M1-M2"]
    sig = [e1 - m2, (e2 - m1) / 1000.0, m1 - m2]  # second channel stored in mV
    m = CleanMontage(eeg=["E1-M2"], eog=[{"E1-M2": 1, "E2-M1": -1, "M1-M2": -1}])
    r = resolve_montage(m, labels, {lab: 100.0 for lab in labels})
    x, fs = read_derivation(_FakeReader(sig, ["uV", "mV", "uV"]), r.derivations["eog"])
    assert fs == 100.0
    np.testing.assert_allclose(x, e1 - e2, atol=1e-9)


def test_fallback_level():
    m = CleanMontage(
        eeg=["C4"], eog=["E"], chin=[Fallback("env", "envelope")], level="poor"
    )
    r = resolve_montage(m, ["C4", "E", "env"], {"C4": 1, "E": 1, "env": 1})
    assert r.level == "envelope"
