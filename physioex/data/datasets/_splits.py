"""Loaders for the published external splits shipped under ``_splits/``.

The ``wav2sleep`` directory holds the train/val/test session-id lists of
Carter & Tarassenko (2024), verbatim from the ``wav2sleep`` repository (MIT
License, see the directory README): ``<dataset>/{train,val,test}.txt`` plus
the census-balanced lists of Jones et al. (2024) in ``census/{val,test}.txt``.
Ids are EDF stems as distributed by NSRR.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

WAV2SLEEP_DIR = Path(__file__).parent / "_splits" / "wav2sleep"
SLEEPPPGNET_REPLICA_DIR = Path(__file__).parent / "_splits" / "sleepppgnet-replica"


def load_wav2sleep_split(
    dataset: str,
    part: str = "train",
    census: bool = False,
) -> List[str]:
    """Return the session ids of one wav2sleep split file, in file order.

    Args:
        dataset: dataset name (``"shhs"``, ``"mesa"``, ``"mros"``, ``"ccshs"``,
            ``"wsc"``, ``"cfs"``, ``"chat"``).
        part: ``"train"``, ``"val"`` or ``"test"``.
        census: if True, read the census-balanced lists (val/test only) from
            ``census/`` instead of the per-dataset directory.
    """
    if census:
        d = WAV2SLEEP_DIR / "census"
    else:
        d = WAV2SLEEP_DIR / dataset
    path = d / f"{part}.txt"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_sleepppgnet_replica_split(dataset: str, part: str = "test") -> List[str]:
    """Return the session ids of the SleepPPG-Net replica split (MESA test only).

    Only ``part="test"`` exists (204 MESA recordings, see the directory
    README); other parts and datasets return an empty list.
    """
    path = SLEEPPPGNET_REPLICA_DIR / dataset / f"{part}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
