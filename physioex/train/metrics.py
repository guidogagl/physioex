import typing
import torch


def accuracy_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    average: str = "micro",
) -> float:
    """Compute accuracy score.

    Args:
        outputs: Model outputs (logits), shape (..., n_classes).
        targets: Ground-truth labels.
        ignore_index: Label value to ignore (default -1).
        average: Averaging method. ``"micro"`` computes global accuracy (correct / total).
            ``"weighted"`` is equivalent to ``"micro"`` for accuracy and is accepted for
            API consistency with the other metric functions.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if average not in ("micro", "weighted"):
        raise ValueError(
            f"accuracy_score 'average' must be 'micro' or 'weighted', got '{average}'"
        )

    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        valid_total = valid_mask.sum()
        if valid_total.item() == 0:
            return 0.0
        correct = (preds[valid_mask] == targets[valid_mask]).float().sum()
        acc = correct / valid_total.float()
    else:
        correct = (preds == targets).float().sum()
        total = torch.tensor(targets.numel(), device=preds.device, dtype=torch.float32)
        acc = correct / total

    return acc.item()


def _per_class_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int):
    """Return per-class F1 scores and supports."""
    f1s = []
    supports = []
    for cls in range(n_classes):
        tp = ((preds == cls) & (targets == cls)).float().sum()
        fp = ((preds == cls) & (targets != cls)).float().sum()
        fn = ((preds != cls) & (targets == cls)).float().sum()
        support = tp + fn  # number of true instances for this class

        if support.item() == 0:
            f1s.append(0.0)
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1s.append(f1.item())
        supports.append(support.item())
    return f1s, supports


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    average: str = "macro",
) -> float:
    """Compute F1 score.

    Args:
        outputs: Model outputs (logits), shape (..., n_classes).
        targets: Ground-truth labels.
        ignore_index: Label value to ignore (default -1).
        average: ``"macro"`` returns unweighted mean of per-class F1 across classes
            with non-zero support. ``"weighted"`` weights each class F1 by its
            support (number of true instances).

    Returns:
        F1 score as a float in [0, 1].
    """
    if average not in ("macro", "weighted"):
        raise ValueError(
            f"f1_score 'average' must be 'macro' or 'weighted', got '{average}'"
        )

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    f1s, supports = _per_class_f1(preds, targets, n_classes)

    if average == "weighted":
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return sum(f * s for f, s in zip(f1s, supports)) / total_support

    # macro: mean over classes with non-zero support
    active = [f for f, s in zip(f1s, supports) if s > 0]
    return sum(active) / len(active) if active else 0.0


def _per_class_precision(preds: torch.Tensor, targets: torch.Tensor, n_classes: int):
    """Return per-class precision scores and supports."""
    precisions = []
    supports = []
    for cls in range(n_classes):
        tp = ((preds == cls) & (targets == cls)).float().sum()
        fp = ((preds == cls) & (targets != cls)).float().sum()
        support = (targets == cls).float().sum()  # true instances

        if support.item() == 0:
            precisions.append(0.0)
        else:
            prec = tp / (tp + fp + 1e-8)
            precisions.append(prec.item())
        supports.append(support.item())
    return precisions, supports


def precision_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    average: str = "macro",
) -> float:
    """Compute precision score.

    Args:
        outputs: Model outputs (logits), shape (..., n_classes).
        targets: Ground-truth labels.
        ignore_index: Label value to ignore (default -1).
        average: ``"macro"`` returns unweighted mean of per-class precision across
            classes with non-zero support. ``"weighted"`` weights each class precision
            by its support.

    Returns:
        Precision as a float in [0, 1].
    """
    if average not in ("macro", "weighted"):
        raise ValueError(
            f"precision_score 'average' must be 'macro' or 'weighted', got '{average}'"
        )

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    precisions, supports = _per_class_precision(preds, targets, n_classes)

    if average == "weighted":
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return sum(p * s for p, s in zip(precisions, supports)) / total_support

    # macro: mean over classes with non-zero support
    active = [p for p, s in zip(precisions, supports) if s > 0]
    return sum(active) / len(active) if active else 0.0


def _per_class_recall(preds: torch.Tensor, targets: torch.Tensor, n_classes: int):
    """Return per-class recall scores and supports."""
    recalls = []
    supports = []
    for cls in range(n_classes):
        tp = ((preds == cls) & (targets == cls)).float().sum()
        fn = ((preds != cls) & (targets == cls)).float().sum()
        support = tp + fn

        if support.item() == 0:
            recalls.append(0.0)
        else:
            rec = tp / (tp + fn + 1e-8)
            recalls.append(rec.item())
        supports.append(support.item())
    return recalls, supports


def recall_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    average: str = "macro",
) -> float:
    """Compute recall score.

    Args:
        outputs: Model outputs (logits), shape (..., n_classes).
        targets: Ground-truth labels.
        ignore_index: Label value to ignore (default -1).
        average: ``"macro"`` returns unweighted mean of per-class recall across classes
            with non-zero support. ``"weighted"`` weights each class recall by its
            support.

    Returns:
        Recall as a float in [0, 1].
    """
    if average not in ("macro", "weighted"):
        raise ValueError(
            f"recall_score 'average' must be 'macro' or 'weighted', got '{average}'"
        )

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    recalls, supports = _per_class_recall(preds, targets, n_classes)

    if average == "weighted":
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return sum(r * s for r, s in zip(recalls, supports)) / total_support

    # macro: mean over classes with non-zero support
    active = [r for r, s in zip(recalls, supports) if s > 0]
    return sum(active) / len(active) if active else 0.0


def cohen_kappa_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    total = targets.numel()
    if total == 0:
        return 0.0

    conf_matrix = torch.zeros(
        (n_classes, n_classes), dtype=torch.float32, device=preds.device
    )
    for t, p in zip(targets.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    po = torch.diag(conf_matrix).sum() / total
    pe = (conf_matrix.sum(dim=0) * conf_matrix.sum(dim=1)).sum() / (total * total)

    kappa = (po - pe) / (1 - pe + 1e-8)

    return kappa.item()


def support_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> list[int]:

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    support_scores = []
    for cls in range(n_classes):
        support = (targets == cls).float().sum()
        support_scores.append(support.item())
    return support_scores


def confusion_matrix(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    normalize: typing.Union[bool, str] = False,
) -> torch.Tensor:

    assert normalize in [
        False,
        True,
        "preds",
        "targets",
    ], "normalize must be one of: False, True, 'preds', 'targets'"

    n_classes = outputs.shape[-1]
    preds = torch.argmax(outputs, dim=-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    conf_matrix = torch.zeros(
        (n_classes, n_classes), dtype=torch.int64, device=preds.device
    )
    for t, p in zip(targets.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    if normalize == "preds":
        conf_matrix = conf_matrix.float() / (
            conf_matrix.sum(dim=0, keepdim=True) + 1e-8
        )
    elif normalize == "targets":
        conf_matrix = conf_matrix.float() / (
            conf_matrix.sum(dim=1, keepdim=True) + 1e-8
        )
    elif normalize == True:
        conf_matrix = conf_matrix.float() / (conf_matrix.sum() + 1e-8)

    return conf_matrix.cpu()


CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1_score": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "cohen_kappa": cohen_kappa_score,
    "confusion_matrix": confusion_matrix,
    "support": support_score,
}


def mse_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """Mean squared error for regression tasks."""
    outputs = outputs.squeeze(-1) if outputs.dim() > targets.dim() else outputs
    if ignore_index is not None:
        mask = targets != ignore_index
        outputs = outputs[mask]
        targets = targets[mask]
    if targets.numel() == 0:
        return 0.0
    return torch.mean((outputs - targets.float()) ** 2).item()


def mae_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """Mean absolute error for regression tasks."""
    outputs = outputs.squeeze(-1) if outputs.dim() > targets.dim() else outputs
    if ignore_index is not None:
        mask = targets != ignore_index
        outputs = outputs[mask]
        targets = targets[mask]
    if targets.numel() == 0:
        return 0.0
    return torch.mean(torch.abs(outputs - targets.float())).item()


def r2_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """Coefficient of determination (R^2) for regression tasks."""
    outputs = outputs.squeeze(-1) if outputs.dim() > targets.dim() else outputs
    if ignore_index is not None:
        mask = targets != ignore_index
        outputs = outputs[mask]
        targets = targets[mask]
    if targets.numel() == 0:
        return 0.0
    targets_f = targets.float()
    ss_res = torch.sum((targets_f - outputs) ** 2)
    ss_tot = torch.sum((targets_f - targets_f.mean()) ** 2)
    if ss_tot.item() == 0.0:
        return 0.0
    return (1 - ss_res / ss_tot).item()


REGRESSION_METRICS = {
    "mse": mse_score,
    "mae": mae_score,
    "r2": r2_score,
}
