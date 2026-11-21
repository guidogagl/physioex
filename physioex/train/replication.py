"""Library support for replicating published cardiorespiratory results.

The three integrated cardiorespiratory sleep-staging networks (Wav2Sleep,
SleepPPG-Net and the ECG-only network of Jones et al.) report their headline
numbers as Cohen's kappa computed in two different ways:

* *pooled* — one confusion matrix over every scored epoch of every recording
  (Wav2Sleep's ``kappa_T``, Jones' ``overall_kappa``);
* *median record* — one kappa per recording, then the median across
  recordings (SleepPPG-Net, Jones' headline number).

This module provides the plumbing to recompute those numbers on whole nights
and check them against the values printed in the papers:

* :func:`iter_nights` pulls whole nights (preprocessed signals + aligned
  labels + metadata) out of a :class:`~physioex.data.base.BasePhysioDataset`
  built in whole-night mode (``sequence_length <= 0``);
* :func:`align` trims a prediction and its label array to a common length the
  way the authors' inference code does (see the docstring);
* :func:`evaluate` wraps
  :func:`physioex.train.metrics.pooled_kappa` and friends;
* :func:`load_paper_metrics` / :func:`check` / :func:`format_table` read the
  paper's claimed values (a small JSON file), compare them with the measured
  ones and render a fixed-width table.

No GPU or data access happens at import time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np

from physioex.train.metrics import (
    median_record_kappa,
    per_record_kappa,
    pooled_kappa,
)

if TYPE_CHECKING:
    from physioex.data.base import BasePhysioDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper targets
# ---------------------------------------------------------------------------


@dataclass
class PaperTarget:
    """One number to replicate from a paper.

    Attributes:
        dataset: dataset name (e.g. ``"shhs"``).
        condition: the split/condition the paper reports (e.g. ``"test"``).
        value: the published value.
        metric: ``"pooled_kappa"`` or ``"median_record_kappa"``.
        tolerance: accepted absolute deviation ``|ours - paper|``.
        assert_: ``True`` = the target must PASS (``FAIL`` otherwise);
            ``False`` = the number is only reported (``REPORT``).
        requires: dataset names whose results must all be present for the
            target to be evaluated (default ``[dataset]``).
        note: free text shown in the table for context.
    """

    dataset: str
    condition: str
    value: float
    metric: str = "pooled_kappa"
    tolerance: float = 0.05
    assert_: bool = True
    requires: List[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.requires:
            self.requires = [self.dataset]
        if self.metric not in ("pooled_kappa", "median_record_kappa"):
            raise ValueError(
                f"metric must be 'pooled_kappa' or 'median_record_kappa', got {self.metric!r}"
            )


def load_paper_metrics(path: "str | Path") -> "tuple[dict, list[PaperTarget]]":
    """Read the paper's claimed results from a JSON file.

    The file has the top-level keys ``"model"``, ``"source"``, ``"metric"``,
    ``"tolerance"`` and ``"targets"`` (a list of objects with keys
    ``dataset``, ``condition``, ``value`` and optional ``metric``,
    ``tolerance``, ``assert`` (bool, default true), ``requires``, ``note``).
    A target's optional ``metric``/``tolerance`` override the top-level ones.

    Args:
        path: path to the JSON file.

    Returns:
        ``(header, targets)`` where ``header`` is the top-level dict without
        ``"targets"`` and ``targets`` the list of :class:`PaperTarget`.
    """
    with open(path) as f:
        doc = json.load(f)
    targets_spec = doc.pop("targets", [])
    default_metric = doc.get("metric", "pooled_kappa")
    default_tolerance = doc.get("tolerance", 0.05)
    targets = [
        PaperTarget(
            dataset=spec["dataset"],
            condition=spec["condition"],
            value=float(spec["value"]),
            metric=spec.get("metric", default_metric),
            tolerance=float(spec.get("tolerance", default_tolerance)),
            assert_=bool(spec.get("assert", True)),
            requires=list(spec.get("requires", [])),
            note=spec.get("note", ""),
        )
        for spec in targets_spec
    ]
    return doc, targets


# ---------------------------------------------------------------------------
# Whole-night iteration
# ---------------------------------------------------------------------------


def iter_nights(
    dataset: "BasePhysioDataset",
    subject_ids: Sequence[str],
) -> "Iterator[tuple[str, dict[str, np.ndarray], np.ndarray, dict]]":
    """Yield one entry per whole night of the given subjects.

    The dataset must be built in whole-night mode (``sequence_length <= 0``);
    in that mode item index ``i`` is exactly subject ``i``
    (see ``BasePhysioDataset._get_recording_item``).

    Args:
        dataset: a ``BasePhysioDataset`` instance (whole-night mode).
        subject_ids: subject ids to iterate over, in any order.

    Yields:
        Tuples ``(subject_id, signals, labels, metadata)`` where

        * ``signals`` maps the channel key the dataset builds items with
          (``channel_order``, e.g. ``"ECG_0"``) to a ``(L, T_e)`` float32
          array of the (preprocessed) whole-night signal;
        * ``labels`` is the ``(L,)`` int64 label array in the dataset's label
          transform (e.g. ``aasm4``: W=0, Light=1, Deep=2, REM=3; ``-1``
          unscored), aligned with the signals;
        * ``metadata`` is the return value of
          ``dataset.subject_metadata(subject_id)`` when the dataset provides
          that method, else ``{}``.
    """
    if dataset.sequence_length > 0:
        raise ValueError(
            "iter_nights needs a dataset in whole-night mode "
            f"(sequence_length <= 0), got sequence_length={dataset.sequence_length}"
        )

    # Recording mode: flat item index i is subject i.
    index = {s.subject_id: i for i, s in enumerate(dataset._subjects)}

    for subject_id in subject_ids:
        if subject_id not in index:
            raise KeyError(f"subject {subject_id!r} not in dataset {dataset.DATASET_NAME}")
        item = dataset[index[subject_id]]
        signals = {
            name: np.asarray(sig, dtype=np.float32)
            for name, sig in item["signals"].items()
        }
        labels = np.asarray(item["labels"], dtype=np.int64)
        try:
            metadata = dataset.subject_metadata(subject_id)
        except AttributeError:
            # not every dataset provides subject-level metadata
            metadata = {}
        yield subject_id, signals, labels, metadata


# ---------------------------------------------------------------------------
# Prediction/label alignment
# ---------------------------------------------------------------------------


def align(
    pred: np.ndarray, labels: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, int]":
    """Cut prediction and labels to their common length.

    Whole-night models with a fixed input grid (Wav2Sleep, SleepPPG-Net: 1200
    epochs) produce predictions for padded epochs when the night is shorter
    than the grid, and the authors' inference drops the tail labels of
    nights longer than the grid. Both effects leave pred and labels at
    different lengths; cutting both to ``min(len)`` recovers the epochs that
    are scored and predicted alike.

    Args:
        pred: prediction array, ``(L_p,)`` class indices (any length).
        labels: label array, ``(L_l,)`` class indices.

    Returns:
        ``(pred_aligned, labels_aligned, n_dropped)`` where the first two
        have shape ``(min(L_p, L_l),)`` and ``n_dropped`` is the number of
        *labelled* epochs that were dropped (``max(0, L_l - min(L_p, L_l))``)
        so callers can report it.
    """
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    n = min(pred.shape[0], labels.shape[0])
    n_dropped = int(max(0, labels.shape[0] - n))
    return pred[:n], labels[:n], n_dropped


# ---------------------------------------------------------------------------
# Kappa evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true_records: "Sequence[np.ndarray]",
    y_pred_records: "Sequence[np.ndarray]",
    n_classes: int,
) -> Dict[str, Any]:
    """Compute the kappa metrics of one evaluation run.

    Wraps :func:`physioex.train.metrics.pooled_kappa`,
    :func:`physioex.train.metrics.median_record_kappa` and
    :func:`physioex.train.metrics.per_record_kappa` (epochs with an unscored
    true label are ignored there).

    Args:
        y_true_records: one label array (class indices, ``-1`` unscored) per
            recording.
        y_pred_records: one prediction array per recording, same order.
        n_classes: number of classes (4 for Wav2Sleep/SleepPPG-Net, 5 for
            Jones).

    Returns:
        dict with ``"pooled_kappa"``, ``"median_record_kappa"``,
        ``"per_record_kappa"`` (list), ``"n_records"`` and ``"n_epochs"``
        (total scored epochs across recordings).
    """
    trues = list(y_true_records)
    preds = list(y_pred_records)
    if len(trues) != len(preds):
        raise ValueError(f"got {len(trues)} true records and {len(preds)} prediction records")
    n_epochs = int(sum(int((t != -1).sum()) for t in trues))
    return {
        "pooled_kappa": pooled_kappa(trues, preds, n_classes),
        "median_record_kappa": median_record_kappa(trues, preds, n_classes),
        "per_record_kappa": per_record_kappa(trues, preds, n_classes).tolist(),
        "n_records": len(trues),
        "n_epochs": n_epochs,
    }


# ---------------------------------------------------------------------------
# Comparison against the paper
# ---------------------------------------------------------------------------


def check(
    results: "dict[tuple[str, str], dict]",
    targets: "Sequence[PaperTarget]",
) -> List[Dict[str, Any]]:
    """Compare measured results against the paper's claimed values.

    Args:
        results: mapping ``(dataset, condition)`` -> the dict returned by
            :func:`evaluate`.
        targets: the paper targets from :func:`load_paper_metrics`.

    Returns:
        One dict per target with keys ``dataset``, ``condition``, ``metric``,
        ``paper``, ``ours``, ``delta``, ``tolerance``, ``status`` (plus
        ``note``). ``status`` is ``"SKIP"`` when any dataset in the target's
        ``requires`` has no results, ``"PASS"``/``"FAIL"`` when
        ``assert_`` is true and ``|delta| <= tolerance`` (or not), and
        ``"REPORT"`` when ``assert_`` is false. ``ours``/``delta`` are
        ``None`` for ``SKIP`` rows.
    """
    rows: List[Dict[str, Any]] = []
    for target in targets:
        row: Dict[str, Any] = {
            "dataset": target.dataset,
            "condition": target.condition,
            "metric": target.metric,
            "paper": target.value,
            "ours": None,
            "delta": None,
            "tolerance": target.tolerance,
            "status": "",
            "note": target.note,
        }
        missing = [
            d for d in target.requires if (d, target.condition) not in results
        ]
        if missing:
            row["status"] = "SKIP"
            row["note"] = (row["note"] + " " if row["note"] else "") + (
                "missing results for: " + ", ".join(missing)
            )
            rows.append(row)
            continue

        # The primary (dataset, condition) result provides the metric.
        result = results[(target.dataset, target.condition)]
        ours = float(result[target.metric])
        delta = ours - target.value
        row["ours"] = ours
        row["delta"] = delta
        if target.assert_:
            row["status"] = "PASS" if abs(delta) <= target.tolerance else "FAIL"
        else:
            row["status"] = "REPORT"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(rows: "Sequence[dict]") -> str:
    """Render comparison rows as a fixed-width text table.

    Args:
        rows: the dicts returned by :func:`check` (any subset of their keys
            is used, in the canonical column order).

    Returns:
        The table as a string (no trailing newline).
    """
    columns = (
        "dataset",
        "condition",
        "metric",
        "paper",
        "ours",
        "delta",
        "tolerance",
        "status",
        "note",
    )
    cells: "list[list[str]]" = []
    for row in rows:
        line = []
        for col in columns:
            value = row.get(col)
            if value is None:
                line.append("-")
            elif isinstance(value, float):
                line.append(f"{value:.4f}")
            else:
                line.append(str(value))
        cells.append(line)

    widths = [len(c) for c in columns]
    for line in cells:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def fmt(line: "Sequence[str]") -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip()

    out = [
        fmt(columns),
        "  ".join("-" * w for w in widths),
    ]
    out.extend(fmt(line) for line in cells)
    return "\n".join(out)
