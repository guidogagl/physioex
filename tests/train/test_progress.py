"""Unit tests for physioex.train.progress (rich progress-bar state).

These exercise the non-TTY code path (pytest captures stdout, so
``Console.is_terminal`` is False and no live display is started). We test the
pure state transitions, not the rendered output.
"""
import pytest

from physioex.train.progress import (
    PhysioExTrainProgressBar,
    PhysioExEvalProgressBar,
)


@pytest.fixture
def bar():
    return PhysioExTrainProgressBar(
        num_epochs=3, steps_per_epoch=10, lr=1e-3, device="cpu"
    )


def test_construct_cpu_disables_device_logging(bar):
    assert bar.log_device is False
    assert bar.num_epochs == 3
    assert bar.steps_per_epoch == 10


def test_best_val_getset(bar):
    bar.set_best_val(acc=0.91, loss=0.42)
    acc, loss = bar.get_best_val()
    assert acc == 0.91 and loss == 0.42


def test_lr_getset(bar):
    bar.set_lr(5e-4)
    assert bar.get_lr() == 5e-4


def test_reset_train_epoch(bar):
    bar.reset_train_epoch(epoch=2)
    assert bar._current_epoch == 2
    assert bar._current_step == 0


def test_update_advances_step_and_records_metrics(bar):
    bar.update(step_loss=0.5, step_acc=0.7, step_time_ms=12.3)
    assert bar._current_step == 1
    assert bar.step_loss == 0.5
    assert bar.step_acc == 0.7
    assert bar.step_time_ms == 12.3


def test_begin_and_end_eval_toggle_state(bar):
    bar.begin_eval(steps_per_eval=5)
    assert bar.show_eval is True
    assert isinstance(bar.eval_progress, PhysioExEvalProgressBar)
    bar.end_eval(val_loss=0.3, val_acc=0.8)
    assert bar.show_eval is False
    assert bar.eval_progress is None


# ---------------------------------------------------------------------------
# Eval progress bar
# ---------------------------------------------------------------------------

def test_eval_bar_update_and_render_line():
    ev = PhysioExEvalProgressBar(steps_per_epoch=4)
    ev.update(step_loss=0.25, step_acc=0.88, step_time_ms=9.0)
    line = ev.render_line()
    assert "0.2500" in line       # loss formatted with 4 decimals
    assert "0.8800" in line       # acc formatted with 4 decimals
    assert "9.0 ms" in line
