"""Unit tests for physioex.train.logger (backend-agnostic logging surface)."""
import argparse

import pytest

from physioex.train.logger import (
    Logger,
    NoOpLogger,
    build_logger,
    add_logger_cli_args,
    logger_train_kwargs,
)


# ---------------------------------------------------------------------------
# ABC + NoOpLogger
# ---------------------------------------------------------------------------

def test_logger_is_abstract():
    with pytest.raises(TypeError):
        Logger()  # abstract .log


def test_noop_logger_accepts_all_hooks():
    log = NoOpLogger()
    # .log is the only required override; the rest default to no-ops.
    assert log.log("train", 0, 0, 1.23, accuracy=0.5, extra_metrics={"f1": 0.4}) is None
    log.log_hparams({"lr": 1e-3}, {"acc": 0.9})
    log.log_learning_rate(0, 1e-3)
    log.log_update_norm(0, 0.1)
    log.log_scalars({"eval/f1": 0.5}, 1)
    log.log_figure("cm", object(), 1)
    log.log_histogram("w", [1, 2, 3], 1)
    log.update()
    log.close()


# ---------------------------------------------------------------------------
# build_logger dispatch
# ---------------------------------------------------------------------------

def test_build_logger_nonzero_rank_is_noop(tmp_path):
    log = build_logger("tensorboard", log_dir=tmp_path, rank=3)
    assert isinstance(log, NoOpLogger)


@pytest.mark.parametrize("kind", ["none", "noop", "off", "false", "NONE"])
def test_build_logger_none_variants(kind):
    assert isinstance(build_logger(kind), NoOpLogger)


def test_build_logger_passthrough_instance():
    existing = NoOpLogger()
    assert build_logger(existing) is existing


def test_build_logger_requires_log_dir_for_real_backend():
    with pytest.raises(ValueError):
        build_logger("tensorboard", log_dir=None)


def test_build_logger_unknown_kind():
    with pytest.raises(ValueError):
        build_logger("bogus", log_dir="/tmp/whatever")


def test_build_logger_tensorboard_degrades_gracefully(tmp_path, monkeypatch):
    """Missing TensorBoard dep must fall back to NoOp with a warning, not crash."""
    import physioex.train.logger as mod

    class _Boom(mod.TensorBoardLogger):
        def __init__(self, *a, **k):
            raise ImportError("no tensorboard")

    monkeypatch.setattr(mod, "TensorBoardLogger", _Boom)
    with pytest.warns(UserWarning):
        log = build_logger("tensorboard", log_dir=tmp_path)
    assert isinstance(log, NoOpLogger)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def test_add_logger_cli_args_and_kwargs_roundtrip():
    parser = argparse.ArgumentParser()
    add_logger_cli_args(parser)
    args = parser.parse_args(["--logger", "wandb", "--run_name", "exp1", "--log_graph"])
    assert args.logger == "wandb"
    assert args.run_name == "exp1"
    assert args.log_graph is True
    assert args.log_confusion_matrix is True  # default on

    kwargs = logger_train_kwargs(args)
    assert kwargs["logger"] == "wandb"
    assert kwargs["run_name"] == "exp1"
    assert kwargs["log_graph"] is True
    assert set(kwargs) >= {
        "logger", "log_dir", "run_name", "tags", "log_graph",
        "log_hist_every", "log_confusion_matrix",
    }


def test_no_confusion_matrix_flag_disables_default():
    parser = argparse.ArgumentParser()
    add_logger_cli_args(parser)
    args = parser.parse_args(["--no_confusion_matrix"])
    assert args.log_confusion_matrix is False


def test_logger_train_kwargs_defaults_on_bare_namespace():
    # getattr fallbacks make this robust to a partial args object.
    kwargs = logger_train_kwargs(argparse.Namespace())
    assert kwargs["logger"] == "tensorboard"
    assert kwargs["log_confusion_matrix"] is True
    assert kwargs["log_hist_every"] == 0
