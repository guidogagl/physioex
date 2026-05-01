"""Benchmark DataLoader configurations on Sleep-EDF.

Measures training throughput (steps/sec) with different num_workers,
pin_memory, and prefetch_factor settings. Uses TinySleepNet (raw pipeline)
as the model — the goal is to measure data loading overhead, not model quality.

Requires Sleep-EDF data and cache to be pre-built.

Usage:
    python examples/pretrained/benchmark_dataloader.py --gpu_id 0
    python examples/pretrained/benchmark_dataloader.py --gpu_id 0 --dataset_root /path/to/sleepedf
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from physioex.data.collate import dict_collate_fn
from physioex.data.datasets import get_dataset
from physioex.models.tinysleepnet import TinySleepNet
from physioex.train.metrics import accuracy_score


N_STEPS = 100
BATCH_SIZE = 32
SEQ_LEN = 20

CONFIGS = [
    {"num_workers": 0, "pin_memory": False, "prefetch_factor": None},
    {"num_workers": 1, "pin_memory": False, "prefetch_factor": 2},
    {"num_workers": 1, "pin_memory": True, "prefetch_factor": 2},
    {"num_workers": 2, "pin_memory": True, "prefetch_factor": 2},
    {"num_workers": 2, "pin_memory": True, "prefetch_factor": 4},
    {"num_workers": 4, "pin_memory": True, "prefetch_factor": 2},
    {"num_workers": 4, "pin_memory": True, "prefetch_factor": 4},
    {"num_workers": 8, "pin_memory": True, "prefetch_factor": 2},
    {"num_workers": 8, "pin_memory": True, "prefetch_factor": 4},
]


def run_benchmark(model, dataset, train_indices, config, device, n_steps):
    """Run n_steps of training with a given DataLoader config and return sec/step."""
    train_subset = Subset(dataset, train_indices)

    loader_kwargs = dict(
        dataset=train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=dict_collate_fn,
    )
    if config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config["prefetch_factor"]
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(**loader_kwargs)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    # Warmup: 5 steps (not timed)
    data_iter = iter(loader)
    for _ in range(min(5, len(loader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        from physioex.data.collate import stack_channels

        inputs = stack_channels(batch).to(device)
        targets = batch["labels"].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(-1, 5), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Timed steps
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    data_iter = iter(loader)
    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inputs = stack_channels(batch).to(device)
        targets = batch["labels"].to(device)

        with torch.autocast(device.type if device.type == "cuda" else "cpu"):
            outputs = model(inputs)

        loss = loss_fn(outputs.reshape(-1, 5), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    # Cleanup workers
    del loader

    return elapsed / n_steps


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader configs")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Dataset
    SleepEDF = get_dataset("sleepedf")
    ds_kwargs = dict(channels=["EEG"], pipelines="raw", sequence_length=SEQ_LEN)
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = SleepEDF(**ds_kwargs)

    train_indices, _, _ = dataset.split(fold=0)
    train_indices = train_indices.tolist()
    print(
        f"Dataset: {dataset.get_n_subjects()} subjects, {len(train_indices)} train samples"
    )

    # Model
    model = TinySleepNet(n_classes=5, in_chan=1).to(device)

    # Benchmark
    print(f"\nBenchmarking {args.n_steps} steps per config, batch_size={BATCH_SIZE}")
    print(
        f"{'workers':>8s} {'pin_mem':>8s} {'prefetch':>9s} {'sec/step':>10s} {'steps/sec':>10s} {'speedup':>8s}"
    )
    print("-" * 60)

    baseline = None
    for config in CONFIGS:
        sec_per_step = run_benchmark(
            model, dataset, train_indices, config, device, args.n_steps
        )

        if baseline is None:
            baseline = sec_per_step
        speedup = baseline / sec_per_step

        pf = str(config["prefetch_factor"]) if config["prefetch_factor"] else "-"
        print(
            f"{config['num_workers']:>8d} "
            f"{str(config['pin_memory']):>8s} "
            f"{pf:>9s} "
            f"{sec_per_step:>10.4f} "
            f"{1.0 / sec_per_step:>10.2f} "
            f"{speedup:>7.2f}x"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
