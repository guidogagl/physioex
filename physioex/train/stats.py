"""Per-subject metric aggregation (mean / std / confidence interval).

The pooled metrics computed by ``Trainer.evaluate`` / ``Trainer.voting_evaluate``
answer "how good is the model overall", but sleep-staging results are reported
per subject: each recording yields one score and we summarise the *distribution*
across subjects (mean +/- std, confidence interval). This module turns a list of
per-subject ``(preds, targets)`` tensors into those aggregates and renders a
confusion-matrix figure for logging.

Standard sleep stage labels: W=0, N1=1, N2=2, N3=3, REM=4.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from physioex.train.metrics import (
    _per_class_f1,
    _per_class_precision,
    _per_class_recall,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Scalar (whole-recording) metrics aggregated across subjects.
SCALAR_METRICS: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "f1_score": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "cohen_kappa": cohen_kappa_score,
}

# Per-class helpers: name -> function returning (values_per_class, supports).
PER_CLASS_METRICS: Dict[str, Callable] = {
    "f1": _per_class_f1,
    "precision": _per_class_precision,
    "recall": _per_class_recall,
}


def _mask_valid(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int):
    argmax = torch.argmax(preds, dim=-1)
    if ignore_index is not None:
        valid = targets != ignore_index
        return argmax[valid], targets[valid]
    return argmax, targets


def per_subject_metrics(
    preds_list: Sequence[torch.Tensor],
    targets_list: Sequence[torch.Tensor],
    n_classes: int,
    ignore_index: int = -1,
) -> Dict[str, List[float]]:
    """Compute scalar and per-class metrics for each subject.

    Parameters
    ----------
    preds_list, targets_list:
        One entry per subject. ``preds`` has shape ``(n_epochs, n_classes)``
        (logits or averaged votes); ``targets`` has shape ``(n_epochs,)``.

    Returns
    -------
    dict mapping a metric key to a list of per-subject values. Scalar metrics
    use their bare name (``"accuracy"``); per-class metrics use
    ``"{metric}/{class_name}"`` (e.g. ``"f1/N2"``).
    """
    class_names = _class_names(n_classes)
    out: Dict[str, List[float]] = {name: [] for name in SCALAR_METRICS}
    for metric in PER_CLASS_METRICS:
        for cname in class_names:
            out[f"{metric}/{cname}"] = []

    for preds, targets in zip(preds_list, targets_list):
        for name, fn in SCALAR_METRICS.items():
            out[name].append(float(fn(preds, targets, ignore_index=ignore_index)))

        argmax, valid_targets = _mask_valid(preds, targets, ignore_index)
        for metric, fn in PER_CLASS_METRICS.items():
            values, _supports = fn(argmax, valid_targets, n_classes)
            for cname, value in zip(class_names, values):
                out[f"{metric}/{cname}"].append(float(value))

    return out


def aggregate(
    per_subject: Dict[str, List[float]],
    ci_method: str = "bootstrap",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-subject values into mean / std / confidence interval.

    Parameters
    ----------
    per_subject:
        Output of :func:`per_subject_metrics` (metric key -> list of values).
    ci_method:
        ``"bootstrap"`` (percentile bootstrap over subjects, robust) or
        ``"normal"`` (mean +/- z * std / sqrt(n)). Falls back to just the mean
        when fewer than two subjects are available.
    n_bootstrap:
        Number of bootstrap resamples (only for ``ci_method="bootstrap"``).
    ci:
        Confidence level (default 0.95).

    Returns
    -------
    dict mapping each metric key to ``{"mean", "std", "ci_low", "ci_high", "n"}``.
    """
    rng = np.random.default_rng(seed)
    z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else _z_for(ci)
    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q

    result: Dict[str, Dict[str, float]] = {}
    for key, values in per_subject.items():
        arr = np.asarray(values, dtype=float)
        n = int(arr.size)
        if n == 0:
            continue

        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0

        if n < 2:
            ci_low = ci_high = mean
        elif ci_method == "normal":
            half = z * std / np.sqrt(n)
            ci_low, ci_high = mean - half, mean + half
        else:  # bootstrap over subjects
            idx = rng.integers(0, n, size=(n_bootstrap, n))
            boot_means = arr[idx].mean(axis=1)
            ci_low = float(np.quantile(boot_means, lower_q))
            ci_high = float(np.quantile(boot_means, upper_q))

        result[key] = {
            "mean": mean,
            "std": std,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n": n,
        }

    return result


def aggregated_scalars(aggregated: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten an :func:`aggregate` result into a scalar mapping for logging.

    Produces keys like ``"eval/f1/N2/mean"`` and ``"eval/f1/N2/std"`` so a
    logger backend can record them via ``log_scalars``.
    """
    flat: Dict[str, float] = {}
    for key, stats in aggregated.items():
        for stat_name in ("mean", "std", "ci_low", "ci_high"):
            flat[f"eval/{key}/{stat_name}"] = stats[stat_name]
    return flat


def confusion_matrix_figure(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_classes: Optional[int] = None,
    ignore_index: int = -1,
    normalize: str | bool = "targets",
    class_names: Optional[Sequence[str]] = None,
):
    """Render a confusion matrix as a matplotlib figure for logging.

    ``normalize="targets"`` gives per-true-class recall rows (the usual sleep
    hypnogram view). Returns a ``matplotlib.figure.Figure``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    cm = confusion_matrix(preds, targets, ignore_index=ignore_index, normalize=normalize)
    return figure_from_cm(cm, class_names=class_names)


def figure_from_cm(cm, class_names: Optional[Sequence[str]] = None):
    """Render an already-computed confusion matrix tensor/array as a figure."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    cm_np = cm.numpy() if isinstance(cm, torch.Tensor) else np.asarray(cm)
    k = cm_np.shape[0]
    names = list(class_names) if class_names is not None else _class_names(k)

    fig, ax = plt.subplots(figsize=(1.4 * k + 1.5, 1.4 * k + 1.0))
    im = ax.imshow(cm_np, cmap="Blues", vmin=0.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    is_float = cm_np.dtype.kind == "f"
    thresh = cm_np.max() / 2.0 if cm_np.size else 0.0
    for i in range(k):
        for j in range(k):
            value = cm_np[i, j]
            text = f"{value:.2f}" if is_float else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig


def _class_names(n_classes: int) -> List[str]:
    if n_classes == len(DEFAULT_CLASS_NAMES):
        return list(DEFAULT_CLASS_NAMES)
    return [f"C{i}" for i in range(n_classes)]


def _z_for(ci: float) -> float:
    """Inverse normal CDF for the two-sided confidence level (no SciPy)."""
    from math import sqrt

    p = 1.0 - (1.0 - ci) / 2.0
    # Beasley-Springer-Moro style rational approximation of the probit.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
