"""Modality classification for PSG channels.

Provides centralized channel-to-modality inference used by both
data pipeline (collate ordering) and models (SleepTokenizer).

Validated on 807 channels across multiple datasets (98.4% accuracy).

Note: ModalityType now has 15 types (added TEMP for temperature sensors).
"""
from enum import IntEnum
import re
from typing import Optional


class ModalityType(IntEnum):
    """14 AASM+ signal types for PSG channels."""

    EEG = 0
    EOG = 1
    EMG = 2
    ECG = 3
    RESP = 4
    SPO2 = 5
    HR = 6
    LEG = 7
    POS = 8
    ACCEL = 9
    LIGHT = 10
    SOUND = 11
    TEMP = 12
    DEVICE = 13
    OTHER = 14

    def __str__(self) -> str:
        return self.name


# Legacy dict for backward compatibility
MODALITY_TYPES = {m.name: i for i, m in enumerate(ModalityType)}
N_MODALITY_TYPES = len(ModalityType)


def infer_channel_modality(channel_name: str, hint: Optional[str] = None) -> ModalityType:
    """Classify a channel name into a modality type.

    Args:
        channel_name: Physical channel label (e.g. ``"EEG C4-M1"``).
        hint: Optional modality hint (one of ``"EEG"``/``"EOG"``/``"EMG"``/``"ECG"`` or ``None``).

    Returns:
        ModalityType enum value.

    Note:
        If hint is provided and valid, it takes precedence over name inference.
        This matches the original behavior from sleep_tokenizer.infer_modality().
    """
    # Hint has precedence (original behavior)
    if hint and hint in MODALITY_TYPES:
        return ModalityType(MODALITY_TYPES[hint])

    n = channel_name.strip()
    u = n.upper()
    c = re.sub(r"[-_\s.()#/]", "", u)

    # ── EEG ──
    if (
        c.startswith("EEG")
        or "CLE" in u
        or "LER" in u
        or c.endswith("REF")
        or c.endswith("AVG")
    ):
        return ModalityType.EEG
    if re.match(r"CH\d+\s*EEG", u):
        return ModalityType.EEG
    eeg_re = (
        r"^(FP[12Z]|AF[34789Z]|F[1-9]0?Z?|FZ|FC[1-6Z]|FT[789]0?|"
        r"C[1-6Z]|CZ|T[3-9]0?|TP[789]0?|CP[1-6Z]|"
        r"P[3489]0?Z?|PZ|PO[3478Z]|O[12Z]|OZ|A[12]|M[12])"
    )
    m = re.match(eeg_re, c)
    if m:
        rest = c[m.end() :]
        if not re.match(
            r"^(ULS|HON|HOD|AP|PG|TT|TL|DS|LM|OS|RES|RESS|ULSE)", rest
        ):
            return ModalityType.EEG

    # ── EOG ──
    if "EOG" in c:
        return ModalityType.EOG
    if re.match(r"^E[12](M[12]|$)", c):
        return ModalityType.EOG
    if c in ("LOC", "ROC", "LEOG", "REOG"):
        return ModalityType.EOG
    # MASS-specific EOG patterns (Left/Right Horiz, Upper/Lower Vertic)
    if re.search(r"HORIZ|VERTIC|HORZ|VERT\.|H\.|V\.", c):
        return ModalityType.EOG

    # ── EMG ──
    if "EMG" in c or "CHIN" in c or "SUBMENTAL" in c:
        return ModalityType.EMG
    if re.search(r"MASSETER|MASSAT|MASRL|SCALENE|SCM$", c):
        return ModalityType.EMG
    if c.startswith("SUBR") or "SUBL" in c:
        return ModalityType.EMG

    # ── ECG ──
    if "ECG" in c or "EKG" in c:
        return ModalityType.ECG
    if re.match(r"^RR\d*$", c):
        return ModalityType.ECG

    # ── LEG ──
    if re.search(r"LEG|TIBIAL|PLM|WPLM", c):
        return ModalityType.LEG
    if re.match(r"^(L|R)?(ARM|FOOT)", c):
        return ModalityType.LEG
    if re.match(r"^(LAT|RAT)\d", c) or c in ("LAT", "RAT"):
        return ModalityType.LEG
    if c.startswith("ARM"):
        return ModalityType.LEG

    # ── RESP ──
    _resp_kw = (
        "FLOW", "NASAL", "NAF", "ORAL", "THOR", "ABDO", "ABD", "CHEST",
        "CANNULA", "THERM", "PTAF", "SNOR", "CPAP", "IPAP", "EPAP", "PAP",
        "LEAK", "LEK", "PRESS", "PRES", "TIDAL", "TIDVOL", "VTOT", "VTINSP",
        "RESP", "RES", "AIRFLOW", "CFLO", "XFLOW", "PFLOW", "SUM", "XSUM",
        "RMI", "WAVE", "PHASE", "CO2", "ETC", "VAB", "VTH",
        "FLATTEN", "DIA", "VENT", "VOLUME", "PUMP", "WINX", "NCPT",
        "EFFORT", "EXOB", "BREATHRATE", "RESPRATE", "NASOR", "NEWAIR", "MV",
    )
    if any(k in c for k in _resp_kw):
        return ModalityType.RESP
    if c in ("NP", "NPV", "TV"):
        return ModalityType.RESP

    # ── SPO2 ──
    if re.search(
        r"SPO2|SAO2|SA02|OXSTAT|OXSTATUS|PLETH|PLTH|PPG|TCPPG|"
        r"PLESMO|NONIN|PULSEAMP|RDPLETH|RDQUALITY",
        c,
    ):
        return ModalityType.SPO2
    if c.startswith("OX"):
        return ModalityType.SPO2

    # ── HR ──
    if c in ("HR", "PULSE", "PR", "DHR", "PULSERATE", "HEARTRATE", "HRATE", "HEART"):
        return ModalityType.HR

    # ── POS ──
    if re.search(r"POS(ITION)?$|BPOS|BODYPOS", c) or c in ("POS", "BODY"):
        return ModalityType.POS

    # ── ACCEL ──
    if re.search(r"ACC|ACCEL|GRAVITY|MOVE|MVMT|ACCU", c):
        return ModalityType.ACCEL

    # ── LIGHT ──
    if re.search(r"LIGHT|PHODB", c):
        return ModalityType.LIGHT

    # ── SOUND ──
    if re.search(r"PHONO|SOUND|MIC", c):
        return ModalityType.SOUND

    # ── TEMP (temperature) ──
    # Body temperature sensors (rectal, skin, oral, etc.)
    if re.search(r"TEMP|THERMISTOR|THERMO|TEMPERATURE", c) and not re.search(
        r"THERM|THERMISTOR", c
    ):
        return ModalityType.TEMP

    # ── DEVICE ──
    if re.search(
        r"STAT$|BUTTN|MARKER|EVENT|BATTERY|ELEV|UNUSED|IMPEDAN|"
        r"PROTECH|GRAPHICAL|^REG\d|^DC\d|^Z1$|^PTT$|^AMPP$|^PDOX$|"
        r"^NDEC$|^PTL$|OTHER|^AUX|^MASK$|RCHON|^PES|^IC[12]|OFF$|"
        r"TECHNICAL|ACTIVITY",
        c,
    ):
        return ModalityType.DEVICE
    if re.match(r"^\d+$", c) or c in (
        "LEFT", "RIGHT", "EPMS", "RIC", "",
    ):
        return ModalityType.DEVICE
    if re.match(r"^[XLRC](\d+)?$", c):
        return ModalityType.DEVICE

    return ModalityType.OTHER
