from rich.console import Console, Group
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.live import Live


try:
    from nvitop import CudaDevice as Device

    _HAS_NVITOP = True
except ImportError:
    _HAS_NVITOP = False
import psutil

from contextlib import contextmanager

# static class to handle training on a single device - single node


class PhysioExTrainProgressBar:
    def __init__(
        self,
        num_epochs: int,
        steps_per_epoch: int,
        lr: float = 1e-3,
        best_val_loss: float = float("inf"),
        best_val_acc: float = 0.0,
        step_loss: float = 0.0,
        step_acc: float = 0.0,
        step_time_ms: float = 0.0,
        device: str = "cpu",
    ):

        self.lr = lr
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.step_loss = step_loss
        self.step_acc = step_acc
        self.step_time_ms = step_time_ms
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self._current_epoch = 0
        self._current_step = 0

        self.live = None
        self.eval_progress = (
            None  # will hold PhysioExEvalProgressBar when eval is active
        )
        self.show_eval = False

        if device == "cpu" or not _HAS_NVITOP:
            self.log_device = False
        elif device == "all":
            self.log_device = True
            self.devices = Device.all()
        else:
            self.log_device = True
            device_id = int(device.replace("cuda:", ""))
            self.devices = [Device.all()[device_id]]

        self.console = Console()
        self._is_tty = self.console.is_terminal

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]Epoch {task.fields[epoch]}/{task.fields[max_epoch]}"
            ),
            BarColumn(),
            TaskProgressColumn(),  # percentuale + completed/total
            TimeElapsedColumn(),  # tempo trascorso per l’epoca
            TimeRemainingColumn(),  # ETA per l’epoca
        )

        self.task = self.progress.add_task(
            description="Train",
            total=self.steps_per_epoch,
            epoch=0,
            max_epoch=self.num_epochs,
        )

        self._start_live()

    def set_best_val(self, acc, loss):
        self.best_val_acc = acc
        self.best_val_loss = loss

    def get_best_val(self):
        return self.best_val_acc, self.best_val_loss

    def set_lr(self, lr: float):
        self.lr = lr

    def get_lr(self) -> float:
        return self.lr

    def reset_train_epoch(self, epoch: int, steps_per_epoch: int = None):
        steps_per_epoch = (
            steps_per_epoch if steps_per_epoch is not None else self.steps_per_epoch
        )
        self._current_epoch = epoch
        self._current_step = 0
        self.progress.reset(
            self.task, total=steps_per_epoch, epoch=epoch, max_epoch=self.num_epochs
        )

    def update(self, step_loss: float, step_acc: float, step_time_ms: float):
        self.step_loss = step_loss
        self.step_acc = step_acc
        self.step_time_ms = step_time_ms
        self._current_step += 1
        self.progress.advance(self.task, 1)
        if self._is_tty:
            self._update_live()
        else:
            # Non-TTY: print a summary line every 10% of the epoch
            interval = max(1, self.steps_per_epoch // 10)
            if self._current_step % interval == 0 or self._current_step == self.steps_per_epoch:
                pct = 100.0 * self._current_step / self.steps_per_epoch
                print(
                    f"[Train] Epoch {self._current_epoch}/{self.num_epochs}  "
                    f"Step {self._current_step}/{self.steps_per_epoch} ({pct:.0f}%)  "
                    f"loss={self.step_loss:.4f}  acc={self.step_acc:.4f}  "
                    f"step_time={self.step_time_ms:.1f}ms  "
                    f"best_val_loss={self.best_val_loss:.4f}  best_val_acc={self.best_val_acc:.2%}",
                    flush=True,
                )

    def begin_eval(self, steps_per_eval: int):
        # create eval progress (no Live inside) and show it under train
        self.eval_progress = PhysioExEvalProgressBar(steps_per_epoch=steps_per_eval)
        self.show_eval = True
        self._update_live()

    def end_eval(self, val_loss: float = None, val_acc: float = None):
        # Non-TTY: print a summary at end of eval
        if not self._is_tty and self.eval_progress is not None:
            # Use averaged values from Trainer if provided, else fall back to last step
            vl = val_loss if val_loss is not None else self.eval_progress.step_loss
            va = val_acc if val_acc is not None else self.eval_progress.step_acc
            print(
                f"[Eval]  Epoch {self._current_epoch}/{self.num_epochs}  "
                f"val_loss={vl:.4f}  val_acc={va:.4f}",
                flush=True,
            )
        self.show_eval = False
        self.eval_progress = None
        self._update_live()

    def _render_header(self):

        device_lines = []
        if self.log_device:
            for dev in self.devices:
                device_lines.append(
                    f"[bold]GPU:[/bold] {dev.name()} | {dev.temperature()}°C | Mem: {dev.memory_used()/ ( 1024 ** 3 ):.2f}/{dev.memory_total() / ( 1024 ** 3 ):.2f} GB | Util: {dev.gpu_utilization()}%"
                )
        # get RAM usage in GB
        ram = psutil.virtual_memory()
        # get system load average
        load1, _, _ = psutil.getloadavg()
        load1 = load1 / psutil.cpu_count()
        device_lines = [
            f"[bold]RAM:[/bold] {ram.used / ( 1024 ** 3 ):.2f}/{ram.total / ( 1024 ** 3 ):.2f} GB | [bold]CPU Load:[/bold] {load1:.2f}%"
        ] + device_lines

        line1 = (
            f"[bold]LR:[/bold] {self.lr:.2e}  |  "
            f"[bold]Best Val. Loss:[/bold] {self.best_val_loss:.4f}  |  "
            f"[bold]Best Val. Acc.:[/bold] {self.best_val_acc:.2%}"
        )
        line2 = (
            f"[bold]Step {self.progress.tasks[0].completed:.0f}/{self.progress.tasks[0].total}[/bold]  |  "
            f"[bold]Step Tr. Loss:[/bold] {self.step_loss:.4f}  |  "
            f"[bold]Step Tr. Acc.:[/bold] {self.step_acc:.2%}  |  "
            f"[bold]Step Time:[/bold] {self.step_time_ms:.1f} ms"
        )
        renderables = device_lines + [line1, line2, self.progress]
        if self.show_eval and self.eval_progress is not None:
            renderables.append(self.eval_progress.render_line())
            renderables.append(self.eval_progress.progress)
        return Group(*renderables)

    def _start_live(self, refresh_per_second: int = 10):
        if not self._is_tty:
            return
        if self.live is None:
            self.live = Live(
                self._render_header(),
                console=self.console,
                refresh_per_second=refresh_per_second,
            )

        self.live.start()

    def _stop_live(self):
        if not self._is_tty:
            return
        assert self.live is not None, "Live not started."
        self.live.stop()
        self.live = None

    def _update_live(self):
        if not self._is_tty or self.live is None:
            return
        self.live.update(self._render_header())


class PhysioExEvalProgressBar:
    def __init__(
        self,
        steps_per_epoch: int,
        step_loss: float = 0.0,
        step_acc: float = 0.0,
        step_time_ms: float = 0.0,
    ):

        self.step_loss = step_loss
        self.step_acc = step_acc
        self.step_time_ms = step_time_ms
        self.steps_per_epoch = steps_per_epoch

        # Pure Progress (no Live). Will be rendered by the train Live.
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Eval"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self.task = self.progress.add_task(
            description="Eval",
            total=self.steps_per_epoch,
        )

    def update(self, step_loss: float, step_acc: float, step_time_ms: float):
        self.step_loss = step_loss
        self.step_acc = step_acc
        self.step_time_ms = step_time_ms
        self.progress.advance(self.task, 1)

    def render_line(self):
        return (
            f"[bold]Step {self.progress.tasks[0].completed:.0f}/{self.progress.tasks[0].total}[/bold]  |  "
            f"[bold]Step Val. Loss:[/bold] {self.step_loss:.4f}  |  "
            f"[bold]Step Val. Acc.:[/bold] {self.step_acc:.4f}  |  "
            f"[bold]Step Time:[/bold] {self.step_time_ms:.1f} ms"
        )
