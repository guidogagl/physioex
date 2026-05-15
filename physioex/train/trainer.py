import os
import typing
import random

import torch
import numpy as np

from torch.utils.data import DataLoader

from time import perf_counter

from physioex.data.dataset import (
    PhysioExDataset,
    _PhysioExTrainDataset,
    _PhysioExEvalDataset,
)

from physioex.train.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    support_score,
)

from physioex.train.progress import PhysioExTrainProgressBar
from physioex.train.losstracker import LossTracker

# import rich for progress bar
from rich.console import Console
from rich.progress import track


def seed_everything(seed: int = 42):
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _BasePhysioEvalDataset(torch.utils.data.Dataset):
    """Wraps a BasePhysioDataset (or MultiDataset) for subject-level evaluation.

    Each ``__getitem__`` returns the **full recording** (entire night) for
    one subject, regardless of the base dataset's ``sequence_length``.
    Used by the Trainer for validation and testing so that evaluation is
    performed per-subject with voting, not per-sequence-window.

    For MultiDataset, ``subject_ids`` are ``(dataset_idx, subject_id)``
    tuples; each subject is looked up in its originating dataset.
    """

    def __init__(self, base_dataset, subject_ids: list):
        self.base = base_dataset
        self.subject_ids = list(subject_ids)
        # Detect MultiDataset
        try:
            from physioex.data.multi import MultiDataset
            self._is_multi = isinstance(base_dataset, MultiDataset)
        except ImportError:
            self._is_multi = False

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        entry = self.subject_ids[idx]

        if self._is_multi:
            # entry is (dataset_idx, subject_id)
            ds_idx, sid = entry
            ds = self.base._datasets[ds_idx]
        else:
            # entry is just subject_id (or (_, subject_id) from split)
            if isinstance(entry, tuple):
                _, sid = entry
            else:
                sid = entry
            ds = self.base

        spec = next(s for s in ds._subjects if s.subject_id == sid)
        n_epochs = ds._n_epochs[sid]
        return ds._build_item(spec, 0, n_epochs)


class Trainer:
    @staticmethod
    def build_dataloaders(
        dataset,  # PhysioExDataset OR BasePhysioDataset
        train_batch_size: int = 32,
        eval_batch_size: int = 1,
        num_workers: int = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        fold: int = 0,
    ) -> tuple[DataLoader, DataLoader]:

        # Detect whether we have the new BasePhysioDataset (dict-returning)
        _is_base_dataset = False
        collate_fn = None
        try:
            from physioex.data.base import BasePhysioDataset as _Base
            from physioex.data.multi import MultiDataset as _Multi

            if isinstance(dataset, (_Base, _Multi)):
                _is_base_dataset = True
                from physioex.data.collate import dict_collate_fn

                collate_fn = dict_collate_fn
                # BasePhysioDataset uses NFS-backed memmap caching; multi-worker
                # DataLoaders cause massive I/O contention on NFS. Default to 0
                # workers (main process) unless the user explicitly requested more.
                if num_workers is None:
                    num_workers = 0
        except ImportError:
            pass

        num_workers = Trainer._get_num_workers(num_workers)

        train_indexes, valid_subjects, test_subjects = dataset.split(fold=fold)

        if _is_base_dataset:
            from torch.utils.data import Subset

            train_dataset = Subset(dataset, train_indexes.tolist())
            # Validation/test: full-recording per subject (for voting evaluation)
            # For MultiDataset, pass (ds_idx, sid) tuples directly;
            # for BasePhysioDataset, extract just the sid.
            try:
                from physioex.data.multi import MultiDataset as _MultiDS
                _is_multi = isinstance(dataset, _MultiDS)
            except ImportError:
                _is_multi = False

            if _is_multi:
                valid_dataset = _BasePhysioEvalDataset(dataset, valid_subjects)
                test_dataset = _BasePhysioEvalDataset(dataset, test_subjects)
            else:
                valid_ids = [sid for _, sid in valid_subjects]
                test_ids = [sid for _, sid in test_subjects]
                valid_dataset = _BasePhysioEvalDataset(dataset, valid_ids)
                test_dataset = _BasePhysioEvalDataset(dataset, test_ids)
        else:
            train_dataset = _PhysioExTrainDataset(dataset, train_indexes)
            valid_dataset = _PhysioExEvalDataset(dataset, valid_subjects)
            test_dataset = _PhysioExEvalDataset(dataset, test_subjects)

        train_loader_kwargs = dict(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if collate_fn is not None:
            train_loader_kwargs["collate_fn"] = collate_fn
        if num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(**train_loader_kwargs)

        # For subject-level eval (BasePhysioDataset), force batch_size=1
        # because recordings have variable lengths and cannot be batched.
        effective_eval_bs = 1 if _is_base_dataset else eval_batch_size
        valid_loader_kwargs = dict(
            dataset=valid_dataset,
            batch_size=effective_eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if collate_fn is not None:
            valid_loader_kwargs["collate_fn"] = collate_fn
        if num_workers > 0:
            valid_loader_kwargs["prefetch_factor"] = prefetch_factor

        valid_loader = DataLoader(**valid_loader_kwargs)

        valid_loader_kwargs["dataset"] = test_dataset
        test_loader = DataLoader(**valid_loader_kwargs)

        return train_loader, valid_loader, test_loader

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        path: str,
        epoch: int = None,
        optimizer: torch.optim.Optimizer = None,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "epoch": epoch,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(
        model: torch.nn.Module,
        path: str,
        optimizer: torch.optim.Optimizer = None,
    ) -> dict:

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint.get("epoch")

        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, epoch, optimizer

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: torch.nn.Module,
        dataset: typing.Union[PhysioExDataset, DataLoader],
        metrics: dict = {
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "cohen_kappa": cohen_kappa_score,
            "confusion_matrix": confusion_matrix,
            "support": support_score,
        },
        batch_size: int = 1,
        num_workers: int = None,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        fold: int = 0,
        gpu_id: int = None,
        ignore_index: int = -1,
        seed: int = 42,
    ) -> dict:

        if seed is not None:
            seed_everything(seed)

        config_params = _get_parameters_from_config()
        batch_size = (
            config_params.get("eval_batch_size", batch_size)
            if config_params.get("eval_batch_size", None) is not None
            else batch_size
        )
        num_workers = (
            config_params.get("num_workers", num_workers)
            if config_params.get("num_workers", None) is not None
            else num_workers
        )
        pin_memory = (
            config_params.get("pin_memory", pin_memory)
            if config_params.get("pin_memory", None) is not None
            else pin_memory
        )
        prefetch_factor = (
            config_params.get("prefetch_factor", prefetch_factor)
            if config_params.get("prefetch_factor", None) is not None
            else prefetch_factor
        )
        persistent_workers = (
            config_params.get("persistent_workers", persistent_workers)
            if config_params.get("persistent_workers", None) is not None
            else persistent_workers
        )
        fold = (
            config_params.get("fold", fold)
            if config_params.get("fold", None) is not None
            else fold
        )
        gpu_id = (
            config_params.get("gpu_id", gpu_id)
            if config_params.get("gpu_id", None) is not None
            else gpu_id
        )

        device = (
            torch.device(f"cuda:{gpu_id}")
            if gpu_id is not None and torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[Info] Using device: {device}")

        # Accept BasePhysioDataset alongside PhysioExDataset and DataLoader
        _is_base_dataset = False
        try:
            from physioex.data.base import BasePhysioDataset as _Base
            from physioex.data.multi import MultiDataset as _Multi

            if isinstance(dataset, (_Base, _Multi)):
                _is_base_dataset = True
        except ImportError:
            pass

        if isinstance(dataset, DataLoader):
            eval_loader = dataset
        elif _is_base_dataset or isinstance(dataset, PhysioExDataset):
            _, _, eval_loader = Trainer.build_dataloaders(
                dataset=dataset,
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                fold=fold,
            )
        else:
            raise ValueError(
                "Invalid dataset type. Must be PhysioExDataset, BasePhysioDataset, or DataLoader."
            )

        model = model.to(device)
        model.eval()

        all_preds, all_targets = [], []

        # show evaluation progress across batches
        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            for batch in track(
                eval_loader,
                description="Evaluating",
                total=len(eval_loader) if hasattr(eval_loader, "__len__") else None,
            ):
                # Support embeddings, dict signals, and legacy tuple batches
                if isinstance(batch, dict) and "embeddings" in batch:
                    inputs = batch["embeddings"].to(device)
                    targets = batch["labels"]
                elif isinstance(batch, dict) and "signals" in batch:
                    from physioex.data.collate import stack_channels

                    inputs = stack_channels(batch).to(device)
                    targets = batch["labels"]
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)

                outputs = model(inputs)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        results = {}
        if metrics is not None:
            for name, metric_fn in metrics.items():
                results[name] = metric_fn(
                    all_preds, all_targets, ignore_index=ignore_index
                )

        return results

    @classmethod
    @torch.no_grad()
    def voting_evaluate(
        cls,
        model: torch.nn.Module,
        dataset: typing.Union[PhysioExDataset, DataLoader],
        L: int = 21,
        metrics: dict = None,
        batch_size: int = 1,
        num_workers: int = None,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        fold: int = 0,
        gpu_id: int = None,
        ignore_index: int = -1,
        seed: int = 42,
    ) -> dict:
        """
        Evaluate using sliding-window voting over full-night sequences.

        For each subject, the night is scanned with L-length windows at L different
        starting offsets (0 to L-1). Predictions from overlapping windows are averaged.
        This matches the evaluation protocol of the current library's voting_strategy().

        Args:
            model: model with forward(x) accepting shape (batch, L, ...) and returning (batch, L, n_classes)
            dataset: a PhysioExDataset (will be wrapped in eval loader) OR a DataLoader
            L: sequence length the model was trained on (default 21)
            metrics: dict of metric_name -> callable(preds, targets, ignore_index=...).
                     Defaults to accuracy/f1/precision/recall/cohen_kappa/confusion_matrix/support.
            ignore_index: label value for padded epochs (default -1)
        Returns:
            dict of metric_name -> value
        """
        if seed is not None:
            seed_everything(seed)

        if metrics is None:
            metrics = {
                "accuracy": accuracy_score,
                "f1_score": f1_score,
                "precision": precision_score,
                "recall": recall_score,
                "cohen_kappa": cohen_kappa_score,
                "confusion_matrix": confusion_matrix,
                "support": support_score,
            }

        config_params = _get_parameters_from_config()
        batch_size = (
            config_params.get("eval_batch_size", batch_size)
            if config_params.get("eval_batch_size", None) is not None
            else batch_size
        )
        num_workers = (
            config_params.get("num_workers", num_workers)
            if config_params.get("num_workers", None) is not None
            else num_workers
        )
        pin_memory = (
            config_params.get("pin_memory", pin_memory)
            if config_params.get("pin_memory", None) is not None
            else pin_memory
        )
        prefetch_factor = (
            config_params.get("prefetch_factor", prefetch_factor)
            if config_params.get("prefetch_factor", None) is not None
            else prefetch_factor
        )
        persistent_workers = (
            config_params.get("persistent_workers", persistent_workers)
            if config_params.get("persistent_workers", None) is not None
            else persistent_workers
        )
        fold = (
            config_params.get("fold", fold)
            if config_params.get("fold", None) is not None
            else fold
        )
        gpu_id = (
            config_params.get("gpu_id", gpu_id)
            if config_params.get("gpu_id", None) is not None
            else gpu_id
        )

        device = (
            torch.device(f"cuda:{gpu_id}")
            if gpu_id is not None and torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[Info] Using device: {device}")

        # Accept BasePhysioDataset alongside PhysioExDataset and DataLoader
        _is_base_dataset_v = False
        try:
            from physioex.data.base import BasePhysioDataset as _Base
            from physioex.data.multi import MultiDataset as _Multi

            if isinstance(dataset, (_Base, _Multi)):
                _is_base_dataset_v = True
        except ImportError:
            pass

        if isinstance(dataset, DataLoader):
            eval_loader = dataset
        elif _is_base_dataset_v or isinstance(dataset, PhysioExDataset):
            _, _, eval_loader = cls.build_dataloaders(
                dataset=dataset,
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                fold=fold,
            )
        else:
            raise ValueError(
                "Invalid dataset type. Must be PhysioExDataset, BasePhysioDataset, or DataLoader."
            )

        model = model.to(device)
        model.eval()

        all_preds, all_targets = [], []

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            for batch in track(
                eval_loader,
                description="Voting evaluation",
                total=len(eval_loader) if hasattr(eval_loader, "__len__") else None,
            ):
                # Support embeddings, dict signals, and legacy tuple batches
                if isinstance(batch, dict) and "embeddings" in batch:
                    inputs = batch["embeddings"].to(device)
                    targets = batch["labels"]
                elif isinstance(batch, dict) and "signals" in batch:
                    from physioex.data.collate import stack_channels

                    inputs = stack_channels(batch).to(device)
                    targets = batch["labels"]  # inputs: (B, L, C, ...), targets: (B, L)
                else:
                    (
                        inputs,
                        targets,
                    ) = batch  # inputs: (B, night_length, ...), targets: (B, night_length)
                inputs = inputs.to(device)
                batch_size_b, night_length = inputs.shape[0], inputs.shape[1]

                if night_length < L:
                    raise ValueError(
                        f"Night length {night_length} is shorter than window L={L}"
                    )

                # Infer n_classes via a single probe forward on the first L epochs
                with torch.no_grad():
                    probe = model(inputs[:, :L])  # (B, L, n_classes)
                n_classes = probe.shape[-1]

                votes = torch.zeros(
                    batch_size_b,
                    night_length,
                    n_classes,
                    device=device,
                    dtype=probe.dtype,
                )
                counts = torch.zeros(
                    batch_size_b, night_length, device=device, dtype=torch.float32
                )

                for offset in range(L):
                    x = inputs[:, offset:]
                    usable = x.shape[1] - (x.shape[1] % L)
                    if usable == 0:
                        continue
                    x = x[:, :usable]
                    num_windows = usable // L
                    # Reshape to (B * num_windows, L, ...)
                    rest_dims = x.shape[2:]
                    x = x.reshape(batch_size_b * num_windows, L, *rest_dims)

                    y = model(x)  # (B * num_windows, L, n_classes)
                    y = y.reshape(batch_size_b, num_windows * L, n_classes)

                    votes[:, offset : offset + usable] += y
                    counts[:, offset : offset + usable] += 1

                # Average votes
                safe_counts = counts.clamp(min=1).unsqueeze(-1)
                votes = votes / safe_counts

                all_preds.append(votes.cpu())
                all_targets.append(targets.cpu())

        # Flatten each subject's predictions to (n_epochs, n_classes) before
        # concatenating, because different subjects have different night lengths.
        flat_preds = [p.reshape(-1, p.shape[-1]) for p in all_preds]
        flat_targets = [t.reshape(-1) for t in all_targets]

        all_preds_cat = torch.cat(flat_preds, dim=0)
        all_targets_cat = torch.cat(flat_targets, dim=0)

        results = {}
        for name, metric_fn in metrics.items():
            results[name] = metric_fn(
                all_preds_cat, all_targets_cat, ignore_index=ignore_index
            )

        return results

    @classmethod
    def train(
        cls,
        model: torch.nn.Module,
        dataset: typing.Union[PhysioExDataset, typing.Tuple[DataLoader, DataLoader]],
        checkpoint_path: str = None,
        max_epochs: int = 10,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(ignore_index=-1),
        optimizer: torch.optim.Optimizer = None,  # Adam by default with lr = 1e-3 weight_decay = 1e-5
        lr: float = 1e-3,  # valid if optimizer is None
        weight_decay: float = 1e-5,  # valid if optimizer is None
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,  # ReduceLROnPlateau by default mode = "min", factor = 0.1, patience = 10
        valid_interval_ratio: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        fold: int = 0,
        gpu_id: int = None,
        log_device: bool = True,
        seed: int = 42,
        accumulate_grad_batches: int = 1,
        early_stopping_patience: int = None,
    ) -> torch.nn.Module:

        if seed is not None:
            seed_everything(seed)

        # load parameters from config file if exists
        config_params = _get_parameters_from_config()
        checkpoint_path = (
            config_params.get("checkpoint_path", checkpoint_path)
            if config_params.get("checkpoint_path", None) is not None
            else checkpoint_path
        )
        train_batch_size = (
            config_params.get("train_batch_size", train_batch_size)
            if config_params.get("train_batch_size", None) is not None
            else train_batch_size
        )
        eval_batch_size = (
            config_params.get("eval_batch_size", eval_batch_size)
            if config_params.get("eval_batch_size", None) is not None
            else eval_batch_size
        )
        num_workers = (
            config_params.get("num_workers", num_workers)
            if config_params.get("num_workers", None) is not None
            else num_workers
        )
        pin_memory = (
            config_params.get("pin_memory", pin_memory)
            if config_params.get("pin_memory", None) is not None
            else pin_memory
        )
        prefetch_factor = (
            config_params.get("prefetch_factor", prefetch_factor)
            if config_params.get("prefetch_factor", None) is not None
            else prefetch_factor
        )
        persistent_workers = (
            config_params.get("persistent_workers", persistent_workers)
            if config_params.get("persistent_workers", None) is not None
            else persistent_workers
        )
        fold = (
            config_params.get("fold", fold)
            if config_params.get("fold", None) is not None
            else fold
        )
        gpu_id = (
            config_params.get("gpu_id", gpu_id)
            if config_params.get("gpu_id", None) is not None
            else gpu_id
        )
        log_device = (
            config_params.get("log_device", log_device)
            if config_params.get("log_device", None) is not None
            else log_device
        )

        if checkpoint_path is None:
            checkpoint_id = 0
            while True:
                checkpoint_path = os.path.join(
                    os.getcwd(), "checkpoints", f"train_{checkpoint_id}"
                )
                # check if directory exists
                if not os.path.exists(checkpoint_path):
                    print(f"[Info] Creating checkpoint directory at {checkpoint_path}")
                    os.makedirs(checkpoint_path)
                    break

                checkpoint_id += 1
        else:
            if not os.path.exists(checkpoint_path):
                print(f"[Info] Creating checkpoint directory at {checkpoint_path}")
                os.makedirs(checkpoint_path)

        device = (
            torch.device(f"cuda:{gpu_id}")
            if gpu_id is not None and torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[Info] Using device: {device}")

        # Accept BasePhysioDataset alongside PhysioExDataset and tuple of DataLoaders
        _is_base_dataset_t = False
        try:
            from physioex.data.base import BasePhysioDataset as _Base
            from physioex.data.multi import MultiDataset as _Multi

            if isinstance(dataset, (_Base, _Multi)):
                _is_base_dataset_t = True
        except ImportError:
            pass

        if isinstance(dataset, tuple):
            train_loader, valid_loader = dataset
        elif _is_base_dataset_t or isinstance(dataset, PhysioExDataset):
            train_loader, valid_loader, _ = cls.build_dataloaders(
                dataset=dataset,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                fold=fold,
            )
        else:
            raise ValueError(
                "Invalid dataset type. Must be PhysioExDataset, BasePhysioDataset, or tuple of DataLoaders."
            )

        valid_interval = max(1, int(len(train_loader) * valid_interval_ratio))

        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            )

        progress = PhysioExTrainProgressBar(
            num_epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            device=str(device) if log_device else "cpu",
        )

        loss_tracker = LossTracker(
            output_dir=checkpoint_path,
            prefix="metrics",
        )

        # Determine sequence_length for voting evaluation
        _eval_seq_len = None
        if _is_base_dataset_t and hasattr(dataset, "sequence_length"):
            _eval_seq_len = dataset.sequence_length

        epochs_without_improvement = 0
        prev_best = float("inf")

        for epoch in range(max_epochs):
            progress.reset_train_epoch(epoch)

            model = cls._run_epoch(
                epoch=epoch,
                model=model,
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                loss=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                valid_interval=valid_interval,
                checkpoint_path=checkpoint_path,
                progress=progress,
                loss_tracker=loss_tracker,
                accumulate_grad_batches=accumulate_grad_batches,
                eval_sequence_length=_eval_seq_len,
            )

            # Early stopping check
            _, current_best_loss = progress.get_best_val()
            if current_best_loss < prev_best:
                prev_best = current_best_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"[Info] Early stopping at epoch {epoch} (patience={early_stopping_patience})"
                )
                break

        progress._stop_live()

        return model

    @classmethod
    def _run_epoch(
        cls,
        epoch: int,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        valid_interval: int,
        checkpoint_path: str,
        progress: PhysioExTrainProgressBar = None,
        loss_tracker: typing.Optional[LossTracker] = None,
        accumulate_grad_batches: int = 1,
        eval_sequence_length : int = None,
    ) -> torch.nn.Module:

        if progress is None:
            stop_progress_at_end = True
            progress = PhysioExTrainProgressBar(
                num_epochs=1, steps_per_epoch=len(train_dataloader), device=str(device)
            )
            progress.reset_train_epoch(epoch)
        else:
            stop_progress_at_end = False

        best_val_acc, best_val_loss = progress.get_best_val()
        train_iter = iter(train_dataloader)
        steps_per_epoch = len(train_dataloader)

        model = model.to(device)
        model.train()

        for step in range(steps_per_epoch):
            t0 = perf_counter()

            step_loss, step_acc, update_norm, step_extra = cls._train_step(
                model=model,
                batch=next(train_iter),
                loss_fn=loss,
                optimizer=optimizer,
                device=device,
                step=step,
                accumulate_grad_batches=accumulate_grad_batches,
            )

            step_time_ms = (perf_counter() - t0) * 1000

            if loss_tracker is not None:
                train_global_step = epoch * steps_per_epoch + step
                loss_tracker.log(
                    stage="train",
                    epoch=epoch,
                    step=train_global_step,
                    loss=step_loss,
                    accuracy=step_acc,
                    extra_metrics=step_extra,
                )
                loss_tracker.log_learning_rate(
                    step=train_global_step,
                    value=optimizer.param_groups[0]["lr"],
                )
                if update_norm is not None:
                    loss_tracker.log_update_norm(
                        step=train_global_step,
                        value=update_norm,
                    )

            progress.update(step_loss, step_acc, step_time_ms)

            if (step + 1) % valid_interval == 0:
                model.eval()

                progress.begin_eval(steps_per_eval=len(valid_dataloader))
                val_iter = iter(valid_dataloader)

                val_losses, val_accs = [], []
                val_extras: list[dict] = []
                for val_step in range(len(valid_dataloader)):
                    v_t0 = perf_counter()

                    if eval_sequence_length is not None:
                        # Subject-level voting evaluation
                        v_step_loss, v_step_acc, v_step_extra = cls._voting_eval_step(
                            model=model,
                            batch=next(val_iter),
                            loss_fn=loss,
                            device=device,
                            L=eval_sequence_length,
                        )
                    else:
                        v_step_loss, v_step_acc, v_step_extra = cls._eval_step(
                            model=model,
                            batch=next(val_iter),
                            loss_fn=loss,
                            device=device,
                        )

                    v_step_time_ms = (perf_counter() - v_t0) * 1000

                    val_losses.append(v_step_loss)
                    val_accs.append(v_step_acc)
                    if v_step_extra is not None:
                        val_extras.append(v_step_extra)

                    progress.eval_progress.update(
                        v_step_loss, v_step_acc, v_step_time_ms
                    )
                    progress._update_live()

                val_loss = sum(val_losses) / len(val_losses)
                val_acc = sum(val_accs) / len(val_accs)

                progress.end_eval(val_loss=val_loss, val_acc=val_acc)

                val_extra_agg = None
                if val_extras:
                    val_extra_agg = {}
                    for metrics in val_extras:
                        for key, value in metrics.items():
                            val_extra_agg[key] = val_extra_agg.get(key, 0.0) + float(
                                value
                            )
                    count = len(val_extras)
                    for key in val_extra_agg:
                        val_extra_agg[key] /= count

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                progress.set_lr(scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr'])

                if loss_tracker is not None:
                    validation_global_step = epoch * steps_per_epoch + step
                    loss_tracker.log(
                        stage="validation",
                        epoch=epoch,
                        step=validation_global_step,
                        loss=val_loss,
                        accuracy=val_acc,
                        extra_metrics=val_extra_agg,
                    )
                    loss_tracker.log_learning_rate(
                        step=validation_global_step,
                        value=scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr'],
                    )
                    loss_tracker.update()

                # aggiorna best su valid ogni tot step (qui dummy)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    progress.set_best_val(best_val_acc, best_val_loss)

                    checkpoint_file = (
                        checkpoint_path
                        + f"/epoch={epoch}-step={step}-val_acc={best_val_acc:.2f}%.pt"
                    )
                    cls.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        path=checkpoint_file,
                    )

                    # if another checkpoint exists remove it (keep only best)
                    for file in os.listdir(checkpoint_path):
                        if file.startswith("epoch=") and file != os.path.basename(
                            checkpoint_file
                        ):
                            os.remove(os.path.join(checkpoint_path, file))

                model.train()

        if stop_progress_at_end:
            progress._stop_live()

        return model

    @classmethod
    def _train_step(
        cls,
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        step: int = 0,
        accumulate_grad_batches: int = 1,
    ) -> tuple[float, float, float | None, dict | None]:

        # Only zero_grad at start of accumulation cycle
        if step % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        with torch.no_grad():
            trainable_params = [
                param for param in model.parameters() if param.requires_grad
            ]
            before_params = [param.detach().clone() for param in trainable_params]

        step_result = cls._step(model, batch, loss_fn, device)
        extra_metrics = None
        if isinstance(step_result, tuple) and len(step_result) == 3:
            loss, acc, extra_metrics = step_result
        else:
            loss, acc = step_result

        # Scale loss to keep equivalent gradient magnitude across accumulation
        scaled_loss = loss / accumulate_grad_batches
        scaled_loss.backward()

        # Only step at end of accumulation cycle
        if (step + 1) % accumulate_grad_batches == 0:
            optimizer.step()

        loss_value = (
            loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
        )
        acc_value = acc.detach().item() if isinstance(acc, torch.Tensor) else float(acc)

        update_norm: float | None
        if trainable_params:
            update_norm_sq = 0.0
            with torch.no_grad():
                for param, param_before in zip(trainable_params, before_params):
                    diff = (param - param_before).view(-1)
                    update_norm_sq += float(torch.dot(diff, diff))

            update_norm = update_norm_sq**0.5 if update_norm_sq > 0.0 else 0.0
        else:
            update_norm = None

        return loss_value, acc_value, update_norm, extra_metrics

    @classmethod
    @torch.no_grad()
    def _eval_step(
        cls,
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float, dict | None]:

        step_result = cls._step(model, batch, loss_fn, device)
        extra_metrics = None
        if isinstance(step_result, tuple) and len(step_result) == 3:
            loss, acc, extra_metrics = step_result
        else:
            loss, acc = step_result

        loss_value = (
            loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
        )
        acc_value = acc.detach().item() if isinstance(acc, torch.Tensor) else float(acc)

        return loss_value, acc_value, extra_metrics

    @classmethod
    @torch.no_grad()
    def _voting_eval_step(
        cls,
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
        L: int = 21,
    ) -> tuple[float, float, dict | None]:
        """Evaluate a single full-night subject batch using sliding-window voting.

        The model was trained with sequence_length=L. For a full-night recording
        of N epochs, we slide L-sized windows at L different offsets, average the
        overlapping predictions, and then compute loss and accuracy on the voted
        output.  This gives a single (loss, accuracy) per subject.
        """
        # Extract inputs and targets from the batch
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels

            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        elif isinstance(batch, dict) and "embeddings" in batch:
            inputs = batch["embeddings"].to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        B, night_length = inputs.shape[0], inputs.shape[1]

        # If the night is shorter than L, fall back to a single forward pass
        if night_length <= L:
            with torch.autocast(device.type if "cuda" in device.type else "cpu"):
                outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = loss_fn(outputs_flat, targets_flat)
            acc = accuracy_score(
                outputs_flat,
                targets_flat,
                ignore_index=getattr(loss_fn, "ignore_index", None),
            )
            return (
                loss.detach().item(),
                acc.detach().item() if isinstance(acc, torch.Tensor) else float(acc),
                None,
            )

        # Probe n_classes from a small forward pass
        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            probe = model(inputs[:, :L])
        n_classes = probe.shape[-1]

        votes = torch.zeros(
            B, night_length, n_classes, device=device, dtype=probe.dtype
        )
        counts = torch.zeros(B, night_length, device=device, dtype=torch.float32)

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            for offset in range(L):
                x = inputs[:, offset:]
                usable = x.shape[1] - (x.shape[1] % L)
                if usable == 0:
                    continue
                x = x[:, :usable]
                num_windows = usable // L
                rest_dims = x.shape[2:]
                x = x.reshape(B * num_windows, L, *rest_dims)

                y = model(x)  # (B*num_windows, L, n_classes)
                y = y.reshape(B, num_windows * L, n_classes)

                votes[:, offset : offset + usable] += y
                counts[:, offset : offset + usable] += 1

        safe_counts = counts.clamp(min=1).unsqueeze(-1)
        voted = votes / safe_counts  # (B, night_length, n_classes)

        voted_flat = voted.reshape(-1, n_classes)
        targets_flat = targets.reshape(-1)

        loss = loss_fn(voted_flat, targets_flat)
        acc = accuracy_score(
            voted_flat,
            targets_flat,
            ignore_index=getattr(loss_fn, "ignore_index", None),
        )

        return (
            loss.detach().item(),
            acc.detach().item() if isinstance(acc, torch.Tensor) else float(acc),
            None,
        )

    @staticmethod
    def _step(
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:

        # Support three batch formats:
        # 1. dict with "embeddings" (EmbeddingDataset for foundation model probes)
        # 2. dict with "signals" (new BasePhysioDataset)
        # 3. tuple (legacy PhysioExDataset)
        if isinstance(batch, dict) and "embeddings" in batch:
            inputs = batch["embeddings"].to(device)
            targets = batch["labels"].to(device)
        elif isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels

            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            outputs = model(inputs)

        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)

        loss = loss_fn(outputs, targets)

        # compute accuracy (dummy here)
        acc = accuracy_score(
            outputs, targets, ignore_index=getattr(loss_fn, "ignore_index", None)
        )

        del inputs, targets, outputs

        return loss, acc

    @staticmethod
    def _get_num_workers(num_workers: int = None) -> int:
        if num_workers is not None:
            return num_workers

        # check if SLURM/TORQUE environment variable exists
        slurm_cpus = os.getenv("SLURM_JOB_CPUS_PER_NODE", None)
        torque_cpus = os.getenv("PBS_NUM_PPN", None)
        if slurm_cpus is not None:
            return int(slurm_cpus)
        elif torque_cpus is not None:
            return int(torque_cpus)
        else:
            return os.cpu_count() if os.cpu_count() is not None else 1


def _get_parameters_from_config():
    import yaml

    config = {}
    try:
        with open("PHYSIOEX_CONFIG.yaml", "r") as f:
            yaml_content = yaml.safe_load(f) or {}
            config = yaml_content.get("Trainer", {})
            print(
                "[Info - Trainer]: Loaded configuration from PHYSIOEX_CONFIG.yaml file."
            )
    except FileNotFoundError:
        print(
            "[Warning - Trainer]: PHYSIOEX_CONFIG.yaml file not found. Using default parameters or those provided in the function call."
        )
    except Exception as exc:
        print(
            f"[Warning - Trainer]: Failed to load PHYSIOEX_CONFIG.yaml due to {exc}. Using default parameters."
        )
    return {
        "checkpoint_path": config.get("checkpoint_path", None),
        "max_epochs": config.get("max_epochs", None),
        "valid_interval_ratio": config.get("valid_interval_ratio", None),
        "train_batch_size": config.get("train_batch_size", None),
        "eval_batch_size": config.get("eval_batch_size", None),
        "num_workers": config.get("num_workers", None),
        "pin_memory": config.get("pin_memory", None),
        "prefetch_factor": config.get("prefetch_factor", None),
        "persistent_workers": config.get("persistent_workers", None),
        "fold": config.get("fold", None),
        "gpu_id": config.get("gpu_id", None),
        "log_device": config.get("log_device", None),
    }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.nn.Linear(128, 5)

    def forward(self, x):
        # x shape batch_size x seqlen x chan x T x F
        batch_size = x.shape[0]
        seqlen = x.shape[1]

        x = x.reshape(batch_size * seqlen, -1)[..., :128]
        x = self.params(x)
        x = x.reshape(batch_size, seqlen, -1)

        return x


if __name__ == "__main__":
    # create dummy model

    model = DummyModel()

    dataset = PhysioExDataset(datasets=["hmc"])

    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=1,
    )

    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
    )

    print(results)
