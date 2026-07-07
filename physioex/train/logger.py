"""Pluggable experiment-tracking loggers for the PhysioEx Trainer.

This module replaces the former ``LossTracker`` (CSV + matplotlib dashboards)
with a small backend-agnostic interface. The training loop talks to a single
``Logger`` object; concrete backends fan the same events out to TensorBoard or
Weights & Biases.

Design notes
------------
- The method surface intentionally mirrors what ``Trainer._run_epoch`` already
  called on the old ``LossTracker`` (``log``, ``log_learning_rate``,
  ``log_update_norm``, ``update``) so the loop change stays minimal, plus the
  richer hooks needed for non-scalar logging (figures, histograms, graph).
- Backend imports (``tensorboard``, ``wandb``) are **lazy**: importing this
  module never requires either dependency. They are only imported when the
  matching backend is instantiated, with a clear error if the ``tracking``
  extra is missing.
- ``build_logger`` returns a ``NoOpLogger`` when ``rank != 0`` so distributed
  training writes metrics from a single process without per-call guards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

Stage = str  # "train" | "validation"


class Logger(ABC):
    """Backend-agnostic experiment logger.

    Every method has a no-op default so backends only override what they
    support; :meth:`log` is the single required hook.
    """

    @abstractmethod
    def log(
        self,
        stage: Stage,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Log the scalar metrics of a single train/validation step."""

    def log_hparams(
        self,
        hparams: Mapping[str, Any],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Record run hyper-parameters once at the start of training."""

    def log_learning_rate(self, step: int, value: float) -> None:
        """Record the learning rate used at a global step."""

    def log_update_norm(self, step: int, value: float) -> None:
        """Record the L2 norm of the optimizer update at a global step."""

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Record a batch of named scalars (per-class metrics, aggregates)."""

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Record a matplotlib figure (e.g. a confusion matrix)."""

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Record a distribution of values (e.g. weights or gradients)."""

    def log_graph(self, model: Any, input_sample: Any) -> None:
        """Record the model computational graph."""

    def watch_model(self, model: Any, log_freq: int = 1000) -> None:
        """Automatically track parameters/gradients (W&B only)."""

    def update(self) -> None:
        """Flush buffered events to the backend."""

    def close(self) -> None:
        """Release backend resources at the end of training."""


class NoOpLogger(Logger):
    """A logger that discards everything.

    Used for ``--logger none``, non-zero ranks in DDP, and tests.
    """

    def log(
        self,
        stage: Stage,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        return None


class TensorBoardLogger(Logger):
    """Log to TensorBoard event files via ``torch.utils.tensorboard``."""

    def __init__(
        self,
        log_dir: Path | str,
        run_name: Optional[str] = None,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "TensorBoard logging requires the 'tracking' extra. "
                "Install it with: pip install 'physioex[tracking]'"
            ) from exc

        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log(
        self,
        stage: Stage,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._writer.add_scalar(f"{stage}/loss", loss, step)
        if accuracy is not None:
            self._writer.add_scalar(f"{stage}/accuracy", accuracy, step)
        if extra_metrics:
            for name, value in extra_metrics.items():
                self._writer.add_scalar(f"{stage}/{name}", float(value), step)

    def log_hparams(
        self,
        hparams: Mapping[str, Any],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        # add_hparams requires JSON-scalar values; coerce everything else to str.
        clean = {
            k: (v if isinstance(v, (int, float, bool, str)) else str(v))
            for k, v in hparams.items()
        }
        try:
            self._writer.add_hparams(clean, dict(metrics or {}))
        except Exception:
            # Fall back to a plain text summary if add_hparams is unhappy.
            text = "\n".join(f"{k}: {v}" for k, v in clean.items())
            self._writer.add_text("hparams", text, 0)

    def log_learning_rate(self, step: int, value: float) -> None:
        self._writer.add_scalar("train/learning_rate", value, step)

    def log_update_norm(self, step: int, value: float) -> None:
        self._writer.add_scalar("train/update_norm", value, step)

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        for name, value in mapping.items():
            self._writer.add_scalar(name, float(value), step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        self._writer.add_figure(tag, figure, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def log_graph(self, model: Any, input_sample: Any) -> None:
        try:
            self._writer.add_graph(model, input_sample)
        except Exception as exc:  # tracing is fragile on dict / dynamic inputs
            self._writer.add_text("graph_error", f"add_graph failed: {exc}", 0)

    def update(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandbLogger(Logger):
    """Log to Weights & Biases.

    Defaults to offline mode (suitable for air-gapped HPC compute nodes); set
    the ``WANDB_MODE`` environment variable to override.
    """

    def __init__(
        self,
        log_dir: Path | str,
        run_name: Optional[str] = None,
        project: str = "physioex",
        tags: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        mode: str = "offline",
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "W&B logging requires the 'tracking' extra. "
                "Install it with: pip install 'physioex[tracking]'"
            ) from exc

        self._wandb = wandb
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=project,
            name=run_name,
            dir=str(log_dir),
            tags=list(tags) if tags else None,
            config=dict(config) if config else None,
            mode=mode,
            reinit=True,
        )

    def log(
        self,
        stage: Stage,
        epoch: int,
        step: int,
        loss: float,
        accuracy: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        payload: dict[str, float] = {f"{stage}/loss": loss, "epoch": epoch}
        if accuracy is not None:
            payload[f"{stage}/accuracy"] = accuracy
        if extra_metrics:
            for name, value in extra_metrics.items():
                payload[f"{stage}/{name}"] = float(value)
        self._wandb.log(payload, step=step)

    def log_hparams(
        self,
        hparams: Mapping[str, Any],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._run.config.update(dict(hparams), allow_val_change=True)

    def log_learning_rate(self, step: int, value: float) -> None:
        self._wandb.log({"train/learning_rate": value}, step=step)

    def log_update_norm(self, step: int, value: float) -> None:
        self._wandb.log({"train/update_norm": value}, step=step)

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self._wandb.log({k: float(v) for k, v in mapping.items()}, step=step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        self._wandb.log({tag: self._wandb.Image(figure)}, step=step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        import numpy as np

        arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
        self._wandb.log({tag: self._wandb.Histogram(arr)}, step=step)

    def log_graph(self, model: Any, input_sample: Any) -> None:
        # W&B captures the graph together with params/grads via watch().
        self.watch_model(model)

    def watch_model(self, model: Any, log_freq: int = 1000) -> None:
        try:
            self._wandb.watch(model, log="all", log_freq=log_freq)
        except Exception:
            pass

    def update(self) -> None:
        return None

    def close(self) -> None:
        self._run.finish()


def add_logger_cli_args(parser) -> None:
    """Register the shared logging CLI flags on an argparse parser."""
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none"],
        help="Experiment-tracking backend.",
    )
    group.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for event files / offline runs (default: <checkpoint>/tb).",
    )
    group.add_argument("--run_name", type=str, default=None, help="Run name.")
    group.add_argument(
        "--tags", nargs="+", default=None, help="Run tags (W&B)."
    )
    group.add_argument(
        "--log_graph",
        action="store_true",
        help="Log the model computational graph.",
    )
    group.add_argument(
        "--log_hist_every",
        type=int,
        default=0,
        help="Log weight/gradient histograms every N steps (0 = off).",
    )
    group.add_argument(
        "--log_confusion_matrix",
        action="store_true",
        default=True,
        help="Log confusion matrix + per-class metrics at validation.",
    )
    group.add_argument(
        "--no_confusion_matrix",
        dest="log_confusion_matrix",
        action="store_false",
        help="Disable confusion-matrix / per-class logging.",
    )
    group.add_argument(
        "--ci_method",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "normal"],
        help="Confidence-interval method for per-subject aggregation.",
    )
    group.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for the CI.",
    )


def logger_train_kwargs(args) -> dict:
    """Extract the ``Trainer.train`` logger keyword-arguments from parsed args."""
    return {
        "logger": getattr(args, "logger", "tensorboard"),
        "log_dir": getattr(args, "log_dir", None),
        "run_name": getattr(args, "run_name", None),
        "tags": getattr(args, "tags", None),
        "log_graph": getattr(args, "log_graph", False),
        "log_hist_every": getattr(args, "log_hist_every", 0),
        "log_confusion_matrix": getattr(args, "log_confusion_matrix", True),
    }


def build_logger(
    kind: str | Logger | None = "tensorboard",
    *,
    log_dir: Optional[Path | str] = None,
    run_name: Optional[str] = None,
    hparams: Optional[Mapping[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
    rank: int = 0,
    project: str = "physioex",
) -> Logger:
    """Instantiate a logger backend.

    Parameters
    ----------
    kind:
        ``"tensorboard"`` (default), ``"wandb"``, ``"none"``, or an already
        constructed :class:`Logger` instance (returned as-is).
    log_dir:
        Directory for event files / offline runs. Required for real backends.
    run_name:
        Human-readable run name (defaults to the log dir basename).
    hparams:
        Hyper-parameters recorded once via :meth:`Logger.log_hparams`.
    tags:
        Optional run tags (used by W&B).
    rank:
        Distributed rank. Any non-zero rank yields a :class:`NoOpLogger`.
    """
    if rank != 0:
        return NoOpLogger()

    if isinstance(kind, Logger):
        return kind

    name = (kind or "none").lower()

    if name in ("none", "noop", "off", "false"):
        return NoOpLogger()

    if log_dir is None:
        raise ValueError(f"build_logger(kind={name!r}) requires a log_dir.")

    if run_name is None:
        run_name = Path(log_dir).name

    if name in ("tensorboard", "tb"):
        # TensorBoard is the default backend, so a missing optional dependency
        # must not crash training: degrade to a NoOpLogger with a warning.
        try:
            logger = TensorBoardLogger(log_dir=log_dir, run_name=run_name)
        except ImportError as exc:
            import warnings

            warnings.warn(
                f"{exc} Falling back to no-op logging. "
                "Install 'physioex[tracking]' to enable TensorBoard."
            )
            return NoOpLogger()
    elif name in ("wandb", "wb"):
        logger = WandbLogger(
            log_dir=log_dir,
            run_name=run_name,
            project=project,
            tags=tags,
            config=hparams,
        )
    else:
        raise ValueError(
            f"Unknown logger '{kind}'. Choose 'tensorboard', 'wandb', or 'none'."
        )

    if hparams:
        logger.log_hparams(hparams)

    return logger
