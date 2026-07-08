import os
import typing

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader

from time import perf_counter

from physioex.data.dataset import (
    PhysioExDataset,
    _PhysioExTrainDataset,
    _PhysioExEvalDataset,
)

from physioex.train.progress import PhysioExTrainProgressBar
from physioex.train.logger import build_logger, Logger
from physioex.train import stats as _stats
from physioex.train.trainer import Trainer as SingleDeviceTrainer

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank: int, world_size: int, device_id: int):
    """
    Args:
         rank: Unique identifier of each process
        world_size: Total number of processes
        device_id: CUDA device index assigned to this rank
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(device_id)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer(SingleDeviceTrainer):
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        path: str,
        epoch: int = None,
        optimizer: torch.optim.Optimizer = None,
    ):
        module = model.module if hasattr(model, "module") else model
        checkpoint = {
            "model_state_dict": module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "epoch": epoch,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def build_dataloaders(
        dataset: PhysioExDataset,
        train_batch_size: int = 32,
        eval_batch_size: int = 1,
        num_workers: int = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        fold: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

        num_workers = Trainer._get_num_workers(num_workers)

        train_indexes, valid_subjects, test_subjects = dataset.split(fold=fold)
        train_dataset = _PhysioExTrainDataset(dataset, train_indexes)
        valid_dataset = _PhysioExEvalDataset(dataset, valid_subjects)
        test_dataset = _PhysioExEvalDataset(dataset, test_subjects)

        train_loader_kwargs = dict(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sampler=DistributedSampler(train_dataset),
        )

        if num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(**train_loader_kwargs)

        valid_loader_kwargs = dict(
            dataset=valid_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            valid_loader_kwargs["prefetch_factor"] = prefetch_factor

        valid_loader = DataLoader(**valid_loader_kwargs)

        valid_loader_kwargs["dataset"] = test_dataset
        test_loader = DataLoader(**valid_loader_kwargs)

        return train_loader, valid_loader, test_loader

    @staticmethod
    def train(
        model: torch.nn.Module,
        dataset: typing.Union[PhysioExDataset, typing.Tuple[DataLoader, DataLoader]],
        checkpoint_path: str = None,
        max_epochs: int = 10,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(ignore_index=-1),
        optimizer: torch.optim.Optimizer = None,  # Adam by default with lr = 1e-3 weight_decay = 1e-5
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,  # ReduceLROnPlateau by default mode = "min", factor = 0.1, patience = 10
        valid_interval_ratio: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        fold: int = 0,
        gpu_ids: typing.Union[str, typing.List[int]] = "all",
        log_device: bool = True,
        logger: str = "tensorboard",
        log_dir: str = None,
        run_name: str = None,
        tags: typing.Optional[typing.List[str]] = None,
        log_graph: bool = False,
        log_hist_every: int = 0,
        log_confusion_matrix: bool = True,
        accumulate_grad_batches: int = 1,
    ) -> torch.nn.Module:

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
        gpu_ids = (
            config_params.get("gpu_ids", gpu_ids)
            if config_params.get("gpu_ids", None) is not None
            else gpu_ids
        )
        log_device = (
            config_params.get("log_device", log_device)
            if config_params.get("log_device", None) is not None
            else log_device
        )
        logger = (
            config_params.get("logger", logger)
            if config_params.get("logger", None) is not None
            else logger
        )
        log_dir = (
            config_params.get("log_dir", log_dir)
            if config_params.get("log_dir", None) is not None
            else log_dir
        )
        run_name = (
            config_params.get("run_name", run_name)
            if config_params.get("run_name", None) is not None
            else run_name
        )
        tags = (
            config_params.get("tags", tags)
            if config_params.get("tags", None) is not None
            else tags
        )
        log_graph = (
            config_params.get("log_graph", log_graph)
            if config_params.get("log_graph", None) is not None
            else log_graph
        )
        log_hist_every = (
            config_params.get("log_hist_every", log_hist_every)
            if config_params.get("log_hist_every", None) is not None
            else log_hist_every
        )
        log_confusion_matrix = (
            config_params.get("log_confusion_matrix", log_confusion_matrix)
            if config_params.get("log_confusion_matrix", None) is not None
            else log_confusion_matrix
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

        if gpu_ids == "all":
            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        else:
            gpu_ids = list(gpu_ids)

        if len(gpu_ids) == 0:
            raise ValueError("No GPU devices available for distributed training.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for multi-device training.")

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            )

        if log_dir is None:
            log_dir = os.path.join(checkpoint_path, "tb")
        if run_name is None:
            run_name = os.path.basename(os.path.normpath(checkpoint_path))

        world_size = len(gpu_ids)
        mp.spawn(
            Trainer.ddp_train_worker,
            args=(
                world_size,
                model,
                dataset,
                checkpoint_path,
                max_epochs,
                loss,
                optimizer,
                scheduler,
                valid_interval_ratio,
                train_batch_size,
                eval_batch_size,
                num_workers,
                pin_memory,
                prefetch_factor,
                persistent_workers,
                fold,
                log_device,
                gpu_ids,
                accumulate_grad_batches,
                logger,
                log_dir,
                run_name,
                tags,
                log_graph,
                log_hist_every,
                log_confusion_matrix,
            ),
            nprocs=world_size,
        )

        # load best model before returning
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
        if checkpoint_files:
            checkpoint_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f)),
                reverse=True,
            )
            model_path = os.path.join(checkpoint_path, checkpoint_files[0])
            model, _, _ = Trainer.load_checkpoint(model, model_path, optimizer=None)
        else:
            print(
                f"[Warning - Trainer]: No checkpoints found in {checkpoint_path}. Returning model without loading state."
            )

        return model

    @classmethod
    def _run_epoch(
        cls,
        rank: int,
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
        logger: typing.Optional[Logger] = None,
        accumulate_grad_batches: int = 1,
        log_hist_every: int = 0,
        log_confusion_matrix: bool = True,
    ) -> torch.nn.Module:

        if progress is None and rank == 0:
            stop_progress_at_end = True
            progress = PhysioExTrainProgressBar(
                num_epochs=1, steps_per_epoch=len(train_dataloader), device=str(device)
            )

            progress.reset_train_epoch(epoch)
        else:
            stop_progress_at_end = False

        if rank == 0:
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

            if rank == 0:
                progress.update(step_loss, step_acc, step_time_ms)

                if logger is not None:
                    train_global_step = epoch * steps_per_epoch + step
                    logger.log(
                        stage="train",
                        epoch=epoch,
                        step=train_global_step,
                        loss=step_loss,
                        accuracy=step_acc,
                        extra_metrics=step_extra,
                    )
                    logger.log_learning_rate(
                        step=train_global_step,
                        value=optimizer.param_groups[0]["lr"],
                    )
                    if update_norm is not None:
                        logger.log_update_norm(
                            step=train_global_step,
                            value=update_norm,
                        )

                    if log_hist_every and train_global_step % log_hist_every == 0:
                        base = model.module if hasattr(model, "module") else model
                        for name, param in base.named_parameters():
                            if not param.requires_grad:
                                continue
                            logger.log_histogram(
                                f"weights/{name}", param.detach(), train_global_step
                            )
                            if param.grad is not None:
                                logger.log_histogram(
                                    f"grads/{name}", param.grad.detach(), train_global_step
                                )

            should_run_eval = (step + 1) % valid_interval == 0

            if should_run_eval:
                loss_tensor = torch.zeros(1, device=device)
                acc_tensor = torch.zeros(1, device=device)

                if rank == 0:
                    model.eval()

                    progress.begin_eval(steps_per_eval=len(valid_dataloader))
                    val_iter = iter(valid_dataloader)

                    collect_val = bool(log_confusion_matrix and logger is not None)

                    val_losses, val_accs = [], []
                    val_extras: list[dict] = []
                    val_preds: list[torch.Tensor] = []
                    val_targets: list[torch.Tensor] = []
                    for val_step in range(len(valid_dataloader)):
                        v_t0 = perf_counter()

                        v_step_loss, v_step_acc, v_step_extra, v_out = cls._eval_step(
                            model=model,
                            batch=next(val_iter),
                            loss_fn=loss,
                            device=device,
                            collect_outputs=collect_val,
                        )

                        v_step_time_ms = (perf_counter() - v_t0) * 1000

                        val_losses.append(v_step_loss)
                        val_accs.append(v_step_acc)
                        if v_step_extra is not None:
                            val_extras.append(v_step_extra)
                        if v_out is not None:
                            val_preds.append(v_out[0])
                            val_targets.append(v_out[1])

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
                                val_extra_agg[key] = val_extra_agg.get(
                                    key, 0.0
                                ) + float(value)
                        count = len(val_extras)
                        for key in val_extra_agg:
                            val_extra_agg[key] /= count

                    loss_tensor.fill_(val_loss)
                    acc_tensor.fill_(val_acc)

                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                    progress.set_lr(scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr'])

                    if logger is not None:
                        validation_global_step = epoch * steps_per_epoch + step
                        logger.log(
                            stage="validation",
                            epoch=epoch,
                            step=validation_global_step,
                            loss=val_loss,
                            accuracy=val_acc,
                            extra_metrics=val_extra_agg,
                        )
                        logger.log_learning_rate(
                            step=validation_global_step,
                            value=scheduler.get_last_lr()[0],
                        )

                        if collect_val and val_preds:
                            ignore_index = getattr(loss, "ignore_index", -1)
                            preds_cat = torch.cat(val_preds, dim=0)
                            targets_cat = torch.cat(val_targets, dim=0)
                            n_classes = preds_cat.shape[-1]

                            argmax, valid_t = _stats._mask_valid(
                                preds_cat, targets_cat, ignore_index
                            )
                            class_names = _stats._class_names(n_classes)
                            per_class = {}
                            for metric, fn in _stats.PER_CLASS_METRICS.items():
                                values, _ = fn(argmax, valid_t, n_classes)
                                for cname, value in zip(class_names, values):
                                    per_class[f"validation/{metric}/{cname}"] = float(value)
                            logger.log_scalars(per_class, validation_global_step)

                            fig = _stats.confusion_matrix_figure(
                                preds_cat, targets_cat, ignore_index=ignore_index
                            )
                            logger.log_figure(
                                "validation/confusion_matrix",
                                fig,
                                validation_global_step,
                            )
                            import matplotlib.pyplot as plt

                            plt.close(fig)

                        logger.update()

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

                        for file in os.listdir(checkpoint_path):
                            if file.startswith("epoch=") and file != os.path.basename(
                                checkpoint_file
                            ):
                                os.remove(os.path.join(checkpoint_path, file))

                    model.train()

                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(loss_tensor, src=0)
                    dist.broadcast(acc_tensor, src=0)

                shared_val_loss = loss_tensor.item()
                shared_val_acc = acc_tensor.item()

                if rank != 0:
                    scheduler.step(shared_val_loss)

        if stop_progress_at_end and rank == 0:
            progress._stop_live()

        return model

    @staticmethod
    def ddp_train_worker(
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        dataset: typing.Union[PhysioExDataset, typing.Tuple[DataLoader, DataLoader]],
        checkpoint_path: str,
        max_epochs: int,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        valid_interval_ratio: float,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
        persistent_workers: bool,
        fold: int,
        log_device: bool,
        gpu_ids: typing.List[int],
        accumulate_grad_batches: int = 1,
        logger_kind: str = "tensorboard",
        log_dir: str = None,
        run_name: str = None,
        tags: typing.Optional[typing.List[str]] = None,
        log_graph: bool = False,
        log_hist_every: int = 0,
        log_confusion_matrix: bool = True,
    ):

        device_id = gpu_ids[rank]
        device = (
            torch.device(f"cuda:{device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        ddp_setup(rank, world_size, device_id)

        if isinstance(dataset, tuple):
            train_loader, valid_loader = dataset
        elif isinstance(dataset, PhysioExDataset):
            train_loader, valid_loader, _ = Trainer.build_dataloaders(
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
                "Invalid dataset type. Must be PhysioExDataset or tuple of DataLoaders."
            )

        valid_interval = max(1, int(len(train_loader) * valid_interval_ratio))

        if rank == 0:
            progress = PhysioExTrainProgressBar(
                num_epochs=max_epochs,
                steps_per_epoch=len(train_loader),
                device="all",
            )
        else:
            progress = None

        # build_logger returns a NoOpLogger for rank != 0, so only rank 0 writes.
        logger = build_logger(
            logger_kind,
            log_dir=log_dir if log_dir is not None else os.path.join(checkpoint_path, "tb"),
            run_name=run_name,
            hparams={
                "model": type(model).__name__,
                "max_epochs": max_epochs,
                "train_batch_size": train_batch_size,
                "world_size": world_size,
            },
            tags=tags,
            rank=rank,
        )

        model = DDP(
            model.to(device), device_ids=[device_id] if device.type == "cuda" else None
        )

        if rank == 0 and log_graph:
            try:
                sample_batch = next(iter(train_loader))
                sample_input = SingleDeviceTrainer._extract_inputs(sample_batch, device)
                logger.log_graph(model, sample_input)
            except Exception as exc:
                print(f"[Warning] Could not log model graph: {exc}")

        for epoch in range(max_epochs):
            if rank == 0:
                progress.reset_train_epoch(epoch)

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            Trainer._run_epoch(
                rank=rank,
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
                logger=logger,
                accumulate_grad_batches=accumulate_grad_batches,
                log_hist_every=log_hist_every,
                log_confusion_matrix=log_confusion_matrix,
            )

        if rank == 0:
            progress._stop_live()

        logger.close()

        destroy_process_group()


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
        "log_device": config.get("log_device", None),
        "gpu_ids": config.get("gpu_ids", None),
        "logger": config.get("logger", None),
        "log_dir": config.get("log_dir", None),
        "run_name": config.get("run_name", None),
        "tags": config.get("tags", None),
        "log_graph": config.get("log_graph", None),
        "log_hist_every": config.get("log_hist_every", None),
        "log_confusion_matrix": config.get("log_confusion_matrix", None),
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

    dataset = PhysioExDataset(datasets=["hmc"])

    Trainer.train(
        model=DummyModel(),
        dataset=dataset,
    )
