"""Per-dataset channel and root path registry for foundation models.

Each dataset declares:
- ``channels``: list of channel requests for BasePhysioDataset
- ``default_root``: dataset path RELATIVE to ``$PHYSIOEX_DATA`` (resolved at
  build time), or ``None`` to let the dataset class derive its own root
- ``channel_map``: raw physioex name → standard 10-20 name
- ``extra_kwargs``: extra constructor args (cohort, visit, subset, etc.)

Channel maps match EEGBenchmarks' adapter CHANNEL_MAP dicts exactly,
ensuring that backbone models receive standard 10-20 channel names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    """Everything needed to create a dataset for foundation model evaluation."""

    channels: List[str]
    # Path to the dataset relative to $PHYSIOEX_DATA (resolved at build time),
    # or None to let the dataset class derive its own root from PHYSIOEX_DATA.
    default_root: Optional[str]
    module_path: str
    class_name: str
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    channel_map: Dict[str, str] = field(default_factory=dict)


# ── Shared channel maps ─────────────────────────────────────────────

_MASS_CHANNEL_MAP = {
    "EEG C3-CLE": "C3",
    "EEG C4-CLE": "C4",
    "EEG Cz-CLE": "Cz",
    "EEG F3-CLE": "F3",
    "EEG F4-CLE": "F4",
    "EEG Fz-CLE": "Fz",
    "EEG O1-CLE": "O1",
    "EEG O2-CLE": "O2",
    "EEG Pz-CLE": "Pz",
    "EEG T3-CLE": "T3",
    "EEG T4-CLE": "T4",
    "EEG T5-CLE": "T5",
    "EEG T6-CLE": "T6",
    "EEG F7-CLE": "F7",
    "EEG F8-CLE": "F8",
    "EEG Fp1-CLE": "Fp1",
    "EEG Fp2-CLE": "Fp2",
    "EEG A2-CLE": "A2",
    "EEG Fpz-CLE": "Fpz",
    "EEG Oz-CLE": "Oz",
    "EEG A1-CLE": "A1",
    "EEG C3-LER": "C3",
    "EEG C4-LER": "C4",
    "EEG Cz-LER": "Cz",
    "EEG F3-LER": "F3",
    "EEG F4-LER": "F4",
    "EEG Fz-LER": "Fz",
    "EEG O1-LER": "O1",
    "EEG O2-LER": "O2",
    "EEG Pz-LER": "Pz",
    "EEG Fp1-LER": "Fp1",
    "EEG Fp2-LER": "Fp2",
    "EEG Oz-LER": "Oz",
    "EEG T3-LER": "T3",
    "EEG T4-LER": "T4",
    "EEG T5-LER": "T5",
    "EEG T6-LER": "T6",
    "EEG F7-LER": "F7",
    "EEG F8-LER": "F8",
}

_ALZHEIMERS_CHANNEL_MAP = {
    "EEG C4-REF": "C4",
    "EEG C3-REF": "C3",
    "EEG F4-REF": "F4",
    "EEG F3-REF": "F3",
    "EEG O2-REF": "O2",
    "EEG O1-REF": "O1",
    "EEG Fz-REF": "Fz",
    "EEG Pz-REF": "Pz",
    "EEG Fp1-REF": "Fp1",
    "EEG Fp2-REF": "Fp2",
    "EEG F7-REF": "F7",
    "EEG F8-REF": "F8",
    "EEG T7-REF": "T7",
    "EEG T8-REF": "T8",
    "EEG P7-REF": "P7",
    "EEG P8-REF": "P8",
}

_PARKINSONS_CHANNEL_MAP = {
    "EEG C3-A2": "C3",
    "EEG C4-A1": "C4",
    "EEG F3-A2": "F3",
    "EEG F4-A1": "F4",
    "EEG O1-A2": "O1",
    "EEG O2-A1": "O2",
    "EEG C3-REF": "C3",
    "EEG C4-REF": "C4",
    "EEG F3-REF": "F3",
    "EEG F4-REF": "F4",
    "EEG O1-REF": "O1",
    "EEG O2-REF": "O2",
}


# ── Dataset configs ──────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # ── Core datasets ───────────────────────────────────────────────
    "hmc": DatasetConfig(
        channels=["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"],
        default_root="hmc/physionet.org/files/hmc-sleep-staging/1.1/recordings",
        module_path="physioex.data.datasets.hmc",
        class_name="HMCDataset",
        channel_map={
            "EEG F4-M1": "F4",
            "EEG C4-M1": "C4",
            "EEG O2-M1": "O2",
            "EEG C3-M2": "C3",
        },
    ),
    "dcsm": DatasetConfig(
        channels=["F4-M1", "C4-M1", "O2-M1", "F3-M2", "C3-M2", "O1-M2"],
        default_root="dcsm/extracted/data/sleep/DCSM",
        module_path="physioex.data.datasets.dcsm",
        class_name="DCSMDataset",
        channel_map={
            "F4-M1": "F4",
            "C4-M1": "C4",
            "O2-M1": "O2",
            "F3-M2": "F3",
            "C3-M2": "C3",
            "O1-M2": "O1",
        },
    ),
    "sleepedf": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="physionet-sleep-data",
        module_path="physioex.data.datasets.sleepedf",
        class_name="SleepEDFDataset",
        channel_map={"EEG Fpz-Cz": "Fpz", "EEG Pz-Oz": "Pz"},
    ),
    # ── MASS cohorts ────────────────────────────────────────────────
    "mass": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 1},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    "mass_ss02": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 2},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    "mass_ss03": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 3},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    "mass_ss04": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 4},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    "mass_ss05": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 5},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    "mass_ss01": DatasetConfig(
        channels=["EEG"] * 18,
        default_root=None,  # Uses PHYSIOEX_DATA env var
        module_path="physioex.data.datasets.mass",
        class_name="MASSDataset",
        extra_kwargs={"cohort": 1},
        channel_map=_MASS_CHANNEL_MAP,
    ),
    # ── NSRR datasets ───────────────────────────────────────────────
    "mesa": DatasetConfig(
        channels=["EEG", "EEG", "EEG"],
        default_root="mesa",
        module_path="physioex.data.datasets.mesa",
        class_name="MESADataset",
        channel_map={"EEG1": "C4", "EEG2": "C3", "EEG3": "Cz"},
    ),
    "mros": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="mros",
        module_path="physioex.data.datasets.mros",
        class_name="MrOSDataset",
        # MrOS resolves to differential pairs (C3,A2) and (C4,A1)
        # which BasePhysioDataset encodes as "C3__A2" and "C4__A1"
        channel_map={"C3__A2": "C3", "C4__A1": "C4"},
    ),
    "shhs_v1": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="shhs",
        module_path="physioex.data.datasets.shhs",
        class_name="SHHSDataset",
        extra_kwargs={"visit": 1},
        channel_map={"EEG": "C4", "EEG(sec)": "C3"},
    ),
    "shhs_v2": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="shhs",
        module_path="physioex.data.datasets.shhs",
        class_name="SHHSDataset",
        extra_kwargs={"visit": 2},
        channel_map={"EEG": "C4", "EEG(sec)": "C3"},
    ),
    # ── WSC visits ──────────────────────────────────────────────────
    "wsc": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="wsc",
        module_path="physioex.data.datasets.wsc",
        class_name="WSCDataset",
        extra_kwargs={"visit": 1},
        # WSC channels resolve to "C3_M2", "C4_M1" → _strip_to_standard handles
    ),
    "wsc_v2": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="wsc",
        module_path="physioex.data.datasets.wsc",
        class_name="WSCDataset",
        extra_kwargs={"visit": 2},
    ),
    "wsc_v3": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="wsc",
        module_path="physioex.data.datasets.wsc",
        class_name="WSCDataset",
        extra_kwargs={"visit": 3},
    ),
    "wsc_v4": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="wsc",
        module_path="physioex.data.datasets.wsc",
        class_name="WSCDataset",
        extra_kwargs={"visit": 4},
    ),
    # ── HPAP variants ───────────────────────────────────────────────
    "hpap_lab_full": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="homepap",
        module_path="physioex.data.datasets.hpap",
        class_name="HPAPDataset",
        extra_kwargs={"subset": "lab-full"},
        # HPAP resolves to differential pairs encoded as "C3__M2", "C4__M1"
        channel_map={"C3__M2": "C3", "C4__M1": "C4"},
    ),
    "hpap_lab_split": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="homepap",
        module_path="physioex.data.datasets.hpap",
        class_name="HPAPDataset",
        extra_kwargs={"subset": "lab-split"},
        channel_map={"C3__M2": "C3", "C4__M1": "C4"},
    ),
    "hpap_home": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="homepap",
        module_path="physioex.data.datasets.hpap",
        class_name="HPAPDataset",
        extra_kwargs={"subset": "home"},
        channel_map={"C3__M2": "C3", "C4__M1": "C4"},
    ),
    # ── STAGES ──────────────────────────────────────────────────────
    "stages": DatasetConfig(
        channels=["EEG"] * 6,
        default_root="stages",
        module_path="physioex.data.datasets.stages",
        class_name="STAGESDataset",
        # STAGES channels are concatenated without separator: "C3M2" not "C3-M2"
        channel_map={
            "C3M2": "C3",
            "C4M1": "C4",
            "F3M2": "F3",
            "F4M1": "F4",
            "O1M2": "O1",
            "O2M1": "O2",
        },
    ),
    # ── Disease datasets ────────────────────────────────────────────
    "alzheimers_ad": DatasetConfig(
        channels=[
            "EEG C4-REF",
            "EEG C3-REF",
            "EEG F4-REF",
            "EEG F3-REF",
            "EEG O2-REF",
            "EEG O1-REF",
            "EEG Fz-REF",
            "EEG Pz-REF",
        ],
        default_root="AlzheimerData",
        module_path="physioex.data.datasets.alzheimers",
        class_name="AlzheimersDataset",
        extra_kwargs={"subset": "AD"},
        channel_map=_ALZHEIMERS_CHANNEL_MAP,
    ),
    "alzheimers_hc": DatasetConfig(
        channels=[
            "EEG C4-REF",
            "EEG C3-REF",
            "EEG F4-REF",
            "EEG F3-REF",
            "EEG O2-REF",
            "EEG O1-REF",
            "EEG Fz-REF",
            "EEG Pz-REF",
        ],
        default_root="AlzheimerData",
        module_path="physioex.data.datasets.alzheimers",
        class_name="AlzheimersDataset",
        extra_kwargs={"subset": "HC"},
        channel_map=_ALZHEIMERS_CHANNEL_MAP,
    ),
    "parkinsons_night_pd": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="Parkinson_data",
        module_path="physioex.data.datasets.parkinsons",
        class_name="ParkinsonsDataset",
        extra_kwargs={"recording": "night", "group": "PD"},
        channel_map=_PARKINSONS_CHANNEL_MAP,
    ),
    "parkinsons_night_hoa": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="Parkinson_data",
        module_path="physioex.data.datasets.parkinsons",
        class_name="ParkinsonsDataset",
        extra_kwargs={"recording": "night", "group": "HOA"},
        channel_map=_PARKINSONS_CHANNEL_MAP,
    ),
    "parkinsons_nap_pd": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="Parkinson_data",
        module_path="physioex.data.datasets.parkinsons",
        class_name="ParkinsonsDataset",
        extra_kwargs={"recording": "nap", "group": "PD"},
        channel_map=_PARKINSONS_CHANNEL_MAP,
    ),
    "parkinsons_nap_hoa": DatasetConfig(
        channels=["EEG", "EEG"],
        default_root="Parkinson_data",
        module_path="physioex.data.datasets.parkinsons",
        class_name="ParkinsonsDataset",
        extra_kwargs={"recording": "nap", "group": "HOA"},
        channel_map=_PARKINSONS_CHANNEL_MAP,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASET_CONFIGS)}"
        )
    return DATASET_CONFIGS[name]


def available_dataset_configs() -> list[str]:
    return sorted(DATASET_CONFIGS.keys())
