"""Named label schemes (hypnogram remappings) for ``BasePhysioDataset``.

Label schemes are callables ``np.ndarray -> np.ndarray`` usable directly as
the ``label_transform`` argument of ``BasePhysioDataset`` (or pass the
scheme *name* -- the dataset resolves it through
:func:`get_label_transform`).

All schemes operate on the AASM 5-class hypnogram (W=0, N1=1, N2=2, N3=3,
REM=4; ``-1`` = unscored) and are vectorised on numpy arrays.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

#: Class names for the AASM 4-class scheme (N1 and N2 merged into "Light"),
#: in target-label order.
AASM4_CLASS_NAMES = ("W", "Light", "Deep", "REM")


def _identity(labels: np.ndarray) -> np.ndarray:
    """AASM 5-class identity scheme: W=0, N1=1, N2=2, N3=3, REM=4, -1 stays -1."""
    return np.asarray(labels)


def _aasm4(labels: np.ndarray) -> np.ndarray:
    """AASM 4-class scheme: W=0, N1->1, N2->1, N3=2, REM=3, -1 stays -1."""
    labels = np.asarray(labels)
    # remap[value + 1] for value in {-1, 0..4}; -1 (unscored) stays -1.
    remap = np.array([-1, 0, 1, 1, 2, 3], dtype=labels.dtype)
    return remap[labels + 1]


_LABEL_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "aasm5": _identity,
    "aasm4": _aasm4,
}


def get_label_transform(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the label transform registered under ``name``.

    Args:
        name: one of ``"aasm5"`` (identity) or ``"aasm4"`` (N1/N2 merged).

    Raises:
        ValueError: if the name is unknown.
    """
    if name not in _LABEL_TRANSFORMS:
        raise ValueError(
            f"Unknown label transform {name!r}. Available: {sorted(_LABEL_TRANSFORMS)}"
        )
    return _LABEL_TRANSFORMS[name]


__all__ = [
    "AASM4_CLASS_NAMES",
    "get_label_transform",
    "available_label_transforms",
]


def available_label_transforms() -> list[str]:
    """Return the sorted list of registered label-transform names."""
    return sorted(_LABEL_TRANSFORMS)
