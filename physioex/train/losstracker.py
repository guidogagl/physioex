"""Utilities for tracking and visualising loss curves during training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Stage = Literal["train", "validation"]


def _normalise_stage(stage: str) -> Stage:
    """Map arbitrary user input to the canonical stage labels."""

    stage_lower = stage.lower()
    if stage_lower in {"train", "training"}:
        return "train"
    if stage_lower in {"val", "valid", "validation", "eval", "evaluation"}:
        return "validation"
    raise ValueError(f"Unsupported stage '{stage}'. Use 'train' or 'validation'.")


@dataclass(slots=True)
class _StageBuffer:
    """Container holding lazy metric buffers for one training stage."""

    loss_epochs: List[int] = field(default_factory=list)
    loss_steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    acc_epochs: List[int] = field(default_factory=list)
    acc_steps: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    extra_steps: Dict[str, List[int]] = field(default_factory=dict)
    extra_values: Dict[str, List[float]] = field(default_factory=dict)

    def append(
        self,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float],
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Store a new measurement without any additional processing."""

        self.loss_epochs.append(epoch)
        self.loss_steps.append(step)
        self.losses.append(loss)
        if accuracy is not None:
            self.acc_epochs.append(epoch)
            self.acc_steps.append(step)
            self.accuracies.append(accuracy)
        if extra_metrics:
            for name, value in extra_metrics.items():
                self.extra_steps.setdefault(name, []).append(step)
                self.extra_values.setdefault(name, []).append(float(value))

    def has_accuracy(self) -> bool:
        """Return True if we recorded any accuracy measurements."""

        return bool(self.accuracies)

    def has_extra_metrics(self) -> bool:
        """Return True if we recorded any extra metrics."""

        return bool(self.extra_values)


class LossTracker:
    """Collect loss metrics and render training curves on demand.

    Example
    -------
    >>> tracker = LossTracker("artifacts")
    >>> tracker.log("train", epoch=0, step=1, loss=0.7, accuracy=0.5)
    >>> tracker.log("validation", epoch=0, step=1, loss=0.8, accuracy=0.4)
    >>> tracker.update()  # doctest: +SKIP
    """

    def __init__(self, output_dir: Path | str, prefix: str = "training") -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self._buffers: Dict[Stage, _StageBuffer] = {
            "train": _StageBuffer(),
            "validation": _StageBuffer(),
        }
        self._lr_steps: List[int] = []
        self._learning_rates: List[float] = []
        self._update_steps: List[int] = []
        self._update_norms: List[float] = []

    def log(
        self,
        stage: str,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Persist a new measurement for later rendering."""

        buffer = self._buffers[_normalise_stage(stage)]
        buffer.append(
            epoch=epoch,
            step=step,
            loss=loss,
            accuracy=accuracy,
            extra_metrics=extra_metrics,
        )

    def log_learning_rate(self, step: int, value: float) -> None:
        """Track the learning rate used at a specific global step."""

        if self._lr_steps and self._lr_steps[-1] == step:
            self._learning_rates[-1] = value
        else:
            self._lr_steps.append(step)
            self._learning_rates.append(value)

    def log_update_norm(self, step: int, value: float) -> None:
        """Track the L2 norm of the optimizer update at a specific step."""

        if self._update_steps and self._update_steps[-1] == step:
            self._update_norms[-1] = value
        else:
            self._update_steps.append(step)
            self._update_norms.append(value)

    def reset(self) -> None:
        """Clear all stored metrics."""

        self._buffers = {
            "train": _StageBuffer(),
            "validation": _StageBuffer(),
        }
        self._lr_steps.clear()
        self._learning_rates.clear()
        self._update_steps.clear()
        self._update_norms.clear()

    def update(self) -> Mapping[str, Path]:
        """Render loss, accuracy, and learning rate plots on a shared canvas."""
        plt.style.use("seaborn-v0_8-darkgrid")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Path] = {}

        self._write_csvs()

        figure_path = self._render_dashboard()
        if figure_path is not None:
            outputs["metrics"] = figure_path

        components_path = self._render_extra_metrics()
        if components_path is not None:
            outputs["loss_components"] = components_path

        return outputs

    def _write_csvs(self) -> None:
        """Write training/validation metrics to CSV files indexed by global step."""

        for stage in ("train", "validation"):
            df = self._build_stage_dataframe(stage)
            if df is None:
                continue
            output_path = self.output_dir / f"{self.prefix}_{stage}.csv"
            df.to_csv(output_path, index=True)

    def _build_stage_dataframe(self, stage: Stage) -> Optional[pd.DataFrame]:
        """Assemble a dataframe with all plotted metrics for a stage."""

        buffer = self._buffers[stage]
        if not buffer.loss_steps and not buffer.acc_steps and not buffer.extra_steps:
            return None

        data: Dict[str, Dict[int, float]] = {}

        if buffer.loss_steps:
            data["loss"] = dict(zip(buffer.loss_steps, buffer.losses))
        if buffer.acc_steps:
            data["accuracy"] = dict(zip(buffer.acc_steps, buffer.accuracies))

        for name, values in buffer.extra_values.items():
            steps = buffer.extra_steps.get(name, [])
            if steps:
                data[name] = dict(zip(steps, values))

        if stage == "train":
            if self._lr_steps:
                data["learning_rate"] = dict(zip(self._lr_steps, self._learning_rates))
            if self._update_steps:
                data["update_norm"] = dict(zip(self._update_steps, self._update_norms))

        if not data:
            return None

        all_steps = sorted({step for series in data.values() for step in series})
        frame = pd.DataFrame(index=all_steps)
        for name, series in data.items():
            frame[name] = pd.Series(series)

        frame.index.name = "global_step"
        return frame

    def _render_dashboard(self) -> Optional[Path]:
        """Render a dashboard with loss, accuracy, and learning-rate curves."""

        train_buffer = self._buffers["train"]
        validation_buffer = self._buffers["validation"]

        has_loss = bool(train_buffer.losses or validation_buffer.losses)
        has_accuracy = bool(train_buffer.accuracies or validation_buffer.accuracies)
        has_lr = bool(self._learning_rates)
        has_updates = bool(self._update_norms)

        if not any([has_loss, has_accuracy, has_lr, has_updates]):
            return None

        fig = plt.figure(figsize=(12, 7))
        grid_spec = fig.add_gridspec(
            2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.25
        )
        ax_loss = fig.add_subplot(grid_spec[0, 0])
        ax_acc = fig.add_subplot(grid_spec[0, 1], sharex=ax_loss)
        ax_lr = fig.add_subplot(grid_spec[1, :], sharex=ax_loss)

        stage_colors = {"train": "C0", "validation": "C1"}

        if has_loss:
            self._plot_stage_series(
                ax=ax_loss,
                values=train_buffer.losses,
                steps=train_buffer.loss_steps,
                color=stage_colors["train"],
                label="Train",
                marker=None,
                smooth_window=50,
                shade_std=True,
            )
            self._plot_stage_series(
                ax=ax_loss,
                values=validation_buffer.losses,
                steps=validation_buffer.loss_steps,
                color=stage_colors["validation"],
                label="Validation",
                marker="o",
                smooth_window=None,
                shade_std=False,
            )
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Loss")
            ax_loss.legend(loc="best")
            ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if has_accuracy:
            self._plot_stage_series(
                ax=ax_acc,
                values=train_buffer.accuracies,
                steps=train_buffer.acc_steps,
                color=stage_colors["train"],
                label="Train",
                marker=None,
                smooth_window=50,
                shade_std=True,
            )
            self._plot_stage_series(
                ax=ax_acc,
                values=validation_buffer.accuracies,
                steps=validation_buffer.acc_steps,
                color=stage_colors["validation"],
                label="Validation",
                marker="o",
                smooth_window=None,
                shade_std=False,
            )
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title("Accuracy")
            ax_acc.legend(loc="best")
            ax_acc.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        lr_lines = []
        if has_lr:
            (line_lr,) = ax_lr.plot(
                self._lr_steps,
                self._learning_rates,
                color="C2",
                linewidth=1.5,
                label="Learning Rate",
            )
            ax_lr.set_ylabel("Learning Rate")
            ax_lr.set_title("Learning Rate / Update Norm")
            ax_lr.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax_lr.set_yscale("log")
            lr_lines.append(line_lr)
        else:
            ax_lr.set_title("Update Norm")
            ax_lr.set_ylabel("Learning Rate")
            ax_lr.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        update_lines = []
        if has_updates:
            ax_update = ax_lr.twinx()
            (line_update,) = ax_update.plot(
                self._update_steps,
                self._update_norms,
                color="C3",
                linewidth=1.4,
                label="Update Norm",
            )
            ax_update.set_ylabel("Update Norm Δθ")
            ax_update.set_yscale("log")
            update_lines.append(line_update)
        else:
            ax_update = None

        if lr_lines or update_lines:
            handles = lr_lines + update_lines
            labels = [line.get_label() for line in handles]
            legend_axis = ax_update if ax_update is not None else ax_lr
            legend_axis.legend(handles, labels, loc="upper right")

        ax_lr.set_xlabel("Global Step")
        # fig.tight_layout()

        output_path = self.output_dir / f"{self.prefix}_metrics.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    def _render_extra_metrics(self) -> Optional[Path]:
        """Render a dashboard with extra logged metrics, if any."""

        train_buffer = self._buffers["train"]
        validation_buffer = self._buffers["validation"]
        metric_names = sorted(
            set(train_buffer.extra_values) | set(validation_buffer.extra_values)
        )

        if not metric_names:
            return None

        cols = 2
        rows = (len(metric_names) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(12, max(4, 3 * rows)), sharey=True
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        stage_colors = {"train": "C0", "validation": "C1"}
        for idx, name in enumerate(metric_names):
            ax = axes[idx]
            if name in train_buffer.extra_values:
                self._plot_stage_series(
                    ax=ax,
                    values=train_buffer.extra_values[name],
                    steps=train_buffer.extra_steps[name],
                    color=stage_colors["train"],
                    label="Train",
                    marker=None,
                    smooth_window=50,
                    shade_std=True,
                )
            if name in validation_buffer.extra_values:
                self._plot_stage_series(
                    ax=ax,
                    values=validation_buffer.extra_values[name],
                    steps=validation_buffer.extra_steps[name],
                    color=stage_colors["validation"],
                    label="Validation",
                    marker="o",
                    smooth_window=None,
                    shade_std=False,
                )
            ax.set_title(name)
            ax.set_xlabel("Global Step")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.legend(loc="best")

        for extra_ax in axes[len(metric_names) :]:
            extra_ax.axis("off")

        output_path = self.output_dir / f"{self.prefix}_loss_components.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    def _plot_stage_series(
        self,
        ax: plt.Axes,
        values: List[float],
        steps: List[int],
        color: str,
        label: str,
        marker: Optional[str],
        smooth_window: Optional[int],
        shade_std: bool,
    ) -> None:
        """Plot a single stage series if values are present."""

        if not values:
            return

        plot_kwargs = {"color": color, "linewidth": 1.5, "label": label}
        if marker is not None:
            plot_kwargs.update(
                {
                    "marker": marker,
                    "markersize": 4,
                    "markerfacecolor": color,
                    "markeredgewidth": 0.0,
                }
            )

        if smooth_window is not None and smooth_window > 1:
            smoothed, deviations = _rolling_mean_std(values, smooth_window)
            ax.plot(steps, smoothed, **plot_kwargs)
            if shade_std and len(smoothed) == len(deviations):
                upper = smoothed + deviations
                lower = smoothed - deviations
                ax.fill_between(
                    steps,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2,
                    linewidth=0.0,
                )
        else:
            ax.plot(steps, values, **plot_kwargs)


def _rolling_mean_std(
    values: List[float], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling statistics using a simple trailing window."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.array([]), np.array([])

    means = np.empty_like(array)
    stds = np.empty_like(array)

    for idx in range(array.size):
        start = max(0, idx - window + 1)
        slice_ = array[start : idx + 1]
        means[idx] = slice_.mean()
        stds[idx] = slice_.std(ddof=0)

    return means, stds
