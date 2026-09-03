"""Synthetic VitalDB fixtures: ``.vital`` files plus the metadata CSVs.

Mirrors :mod:`tests.factories.edf` for the VitalRecorder container.  Requires
the optional ``vitaldb`` dependency; tests should guard with
``pytest.importorskip("vitaldb")``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Columns of VitalDB's cases.csv that the dataset layer touches.  The real table
# has 74; only these matter here.
CASE_COLUMNS = [
    "caseid",
    "subjectid",
    "ane_type",
    "age",
    "sex",
    "asa",
    "icu_days",
    "death_inhosp",
    "opend",
    "dis",
]

T0 = 1_600_000_000.0  # arbitrary but fixed POSIX start time


def write_fake_vital(
    path: Path,
    duration_sec: float = 600.0,
    fs: float = 128.0,
    track_names: Sequence[str] = ("BIS/EEG1_WAV", "BIS/EEG2_WAV"),
    numeric_tracks: Sequence[str] = ("BIS/BIS", "BIS/SR"),
    unit: str = "uV",
    seed: int = 0,
) -> None:
    """Write a tiny ``.vital`` file with waveform and numeric tracks.

    Waveforms are Gaussian noise (20 µV) plus a 10 Hz sine, so band-selective
    preprocessing is observable.  Numeric tracks carry ``srate=0`` and exist to
    check that the header probe excludes them from the channel list.
    """
    from vitaldb import VitalFile

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(fs * duration_sec)
    chunk = int(fs)  # one record per second

    vf = VitalFile()
    vf.dtstart, vf.dtend = T0, T0 + duration_sec
    vf.add_device("BIS", "BIS")

    t = np.arange(n) / fs
    for k, name in enumerate(track_names):
        sig = (20.0 * rng.standard_normal(n) + 10.0 * np.sin(2 * np.pi * 10 * t + k)).astype(
            np.float32
        )
        recs = [
            {"dt": T0 + i / fs, "val": sig[i : i + chunk]} for i in range(0, n, chunk)
        ]
        vf.add_track(name, recs, srate=fs, unit=unit)

    for name in numeric_tracks:
        recs = [{"dt": T0 + s, "val": float(50 + s % 10)} for s in range(int(duration_sec))]
        vf.add_track(name, recs, srate=0, unit="%")

    vf.save_vital(str(path))


def write_fake_meta(
    root: Path,
    cases: Iterable[Dict[str, object]],
    tracks_per_case: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Write ``meta/cases.csv`` and ``meta/trks.csv`` under *root*.

    Args:
        cases: dicts with at least ``caseid``; missing columns are defaulted.
        tracks_per_case: ``{caseid: [track names]}``.  Defaults to both EEG
            channels for every case.
    """
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        row = {c: "" for c in CASE_COLUMNS}
        row.update({"ane_type": "General", "age": "60", "sex": "M", "asa": "2"})
        row.update({k: str(v) for k, v in case.items()})
        row.setdefault("subjectid", row["caseid"])
        if not row["subjectid"]:
            row["subjectid"] = row["caseid"]
        rows.append(row)

    with open(meta / "cases.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if tracks_per_case is None:
        tracks_per_case = {r["caseid"]: ["BIS/EEG1_WAV", "BIS/EEG2_WAV"] for r in rows}

    with open(meta / "trks.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["caseid", "tname", "tid"])
        for caseid, names in tracks_per_case.items():
            for i, name in enumerate(names):
                writer.writerow([caseid, name, f"{caseid}-{i}"])


def build_fake_vitaldb(
    root: Path,
    n_cases: int = 3,
    duration_sec: float = 600.0,
    fs: float = 128.0,
) -> Path:
    """Create a complete miniature VitalDB tree and return *root*."""
    (root / "raw").mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(1, n_cases + 1):
        cid = str(i)
        write_fake_vital(root / "raw" / f"{cid}.vital", duration_sec=duration_sec, fs=fs, seed=i)
        cases.append(
            {
                "caseid": cid,
                "subjectid": str((i + 1) // 2),  # cases 1&2 share a patient
                "icu_days": "3" if i % 2 else "0",
                "death_inhosp": "0",
            }
        )
    write_fake_meta(root, cases)
    return root
