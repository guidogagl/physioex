"""End-to-end smoke tests: real models + new dataset pipeline.

Verifies:
  1. TinySleepNet trains 1 epoch on HMC-like fake data with time_domain preset
  2. SeqSleepNet trains 1 epoch on HMC-like fake data with time_frequency preset
  3. Both models' loss decreases (or at least doesn't crash)
  4. Checkpoints are saved
  5. Trainer.evaluate works with dict batches

Uses synthetic fake EDF (no real data / GPU required).
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.factories.edf import (
    write_fake_edf,
    write_fake_annotations_edf,
    FakeEDFDataset,
)
from physioex.data.collate import dict_collate_fn, stack_channels
from physioex.data.presets import get_preset

# Models
from physioex.models.tinysleepnet import TinySleepNet
from physioex.models.seqsleepnet import SeqSleepNet

# Trainer
from physioex.train.trainer import Trainer

# ---------------------------------------------------------------------------
# Test bookkeeping
# ---------------------------------------------------------------------------
def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# Helpers: create fake data directory
# ---------------------------------------------------------------------------

def create_fake_data(data_dir: Path, n_subjects: int = 3,
                     duration_sec: float = 600.0, fs: int = 100):
    """Create fake EDF files for multiple subjects.

    600s = 20 epochs of 30s each. Stages cycle W,N1,N2,N3,R x 4.
    3 channels: C4-M2, EOG, EMG at 100Hz.
    """
    stages = ["W", "N1", "N2", "N3", "R"] * 4  # 20 stages for 20 epochs
    n_epochs = int(duration_sec / 30.0)
    stages = stages[:n_epochs]

    channel_names = ["C4-M2", "EOG", "EMG"]
    n_channels = len(channel_names)

    for i in range(n_subjects):
        sid = f"SUB{i:02d}"
        write_fake_edf(
            data_dir / f"{sid}.edf",
            n_channels=n_channels,
            duration_sec=duration_sec,
            fs_per_channel=[fs] * n_channels,
            channel_names=channel_names,
            seed=i,
        )
        write_fake_annotations_edf(
            data_dir / f"{sid}_sleepscoring.edf",
            stages=stages,
        )


class MultiSubjectFakeEDFDataset(FakeEDFDataset):
    """FakeEDFDataset variant that supports multiple subjects."""

    def __init__(self, root, subject_ids, **kwargs):
        self._subject_ids = subject_ids
        # Call BasePhysioDataset.__init__ through FakeEDFDataset,
        # but we need to bypass FakeEDFDataset's single-subject assumption.
        # We override _list_subjects instead.
        self._fake_subject_id = subject_ids[0]  # needed by parent __init__
        super().__init__(root=root, subject_id=subject_ids[0], **kwargs)

    def _list_subjects(self):
        from physioex.data.base import SubjectSpec
        root = Path(self.root)
        return [
            SubjectSpec(
                subject_id=sid,
                edf_path=root / f"{sid}.edf",
                label_path=root / f"{sid}_sleepscoring.edf",
            )
            for sid in self._subject_ids
        ]


def make_dataloaders(dataset, batch_size=2, num_workers=0, fold=0):
    """Build train + valid DataLoaders from a BasePhysioDataset.

    Uses dict_collate_fn so the Trainer's _step gets dict batches.
    """
    train_indices, valid_subjects, test_subjects = dataset.split(fold=fold)

    # Train: subset by flat indices
    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
    )

    # Valid: for simplicity, use the full dataset (all indices) as validation.
    # In a real scenario we'd build per-subject loaders, but for a smoke test
    # any data will do.
    all_indices = list(range(len(dataset)))
    valid_subset = torch.utils.data.Subset(dataset, all_indices[:max(1, len(all_indices) // 3)])
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
    )

    return train_loader, valid_loader


# ---------------------------------------------------------------------------
# Test 0: Shape sanity checks
# ---------------------------------------------------------------------------

def test_shape_sanity_time_domain():
    """Verify that time_domain preset produces correct shapes for stacking."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        create_fake_data(data_dir, n_subjects=1, duration_sec=300.0)

        ds = MultiSubjectFakeEDFDataset(
            root=str(data_dir),
            subject_ids=["SUB00"],
            channels=["EEG", "EOG", "EMG"],
            pipelines="time_domain",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        item = ds[0]
        # Collate two items
        batch = dict_collate_fn([ds[0], ds[1]])
        tensor = stack_channels(batch)

        # Expected: (2, 5, 3, 3000) -- B=2, L=5, C=3, T=3000
        expected_shape = (2, 5, 3, 3000)
        ok = tensor.shape == expected_shape
        report(
            "time_domain shape sanity",
            ok,
            f"got {tuple(tensor.shape)}, expected {expected_shape}",
        )
        return ok


def test_shape_sanity_time_frequency():
    """Verify that time_frequency preset produces correct shapes for stacking."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        create_fake_data(data_dir, n_subjects=1, duration_sec=300.0)

        ds = MultiSubjectFakeEDFDataset(
            root=str(data_dir),
            subject_ids=["SUB00"],
            channels=["EEG", "EOG", "EMG"],
            pipelines="time_frequency",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        item = ds[0]
        batch = dict_collate_fn([ds[0], ds[1]])
        tensor = stack_channels(batch)

        # Expected: (2, 5, 3, 29, 129) -- B=2, L=5, C=3, T=29, F=129
        expected_shape = (2, 5, 3, 29, 129)
        ok = tensor.shape == expected_shape
        report(
            "time_frequency shape sanity",
            ok,
            f"got {tuple(tensor.shape)}, expected {expected_shape}",
        )
        return ok


# ---------------------------------------------------------------------------
# Test 1: TinySleepNet + time_domain
# ---------------------------------------------------------------------------

def test_tinysleepnet_time_domain():
    """Train TinySleepNet for 1 epoch on fake data with time_domain preset."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir, \
         tempfile.TemporaryDirectory() as ckpt_dir:
        data_dir = Path(data_dir)
        create_fake_data(data_dir, n_subjects=3, duration_sec=600.0)

        ds = MultiSubjectFakeEDFDataset(
            root=str(data_dir),
            subject_ids=["SUB00", "SUB01", "SUB02"],
            channels=["EEG", "EOG", "EMG"],
            pipelines="time_domain",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        model = TinySleepNet(n_classes=5, in_chan=3, sf=100)

        train_loader, valid_loader = make_dataloaders(
            ds, batch_size=2, num_workers=0, fold=0,
        )

        print(f"\n  [info] TinySleepNet: train_loader has {len(train_loader)} batches, "
              f"valid_loader has {len(valid_loader)} batches")

        trained_model = Trainer.train(
            model=model,
            dataset=(train_loader, valid_loader),
            checkpoint_path=ckpt_dir,
            max_epochs=1,
            lr=1e-3,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            gpu_id=None,
            seed=42,
        )

        # Verify: model returned
        ok_model = trained_model is not None
        # Verify: checkpoint file was saved
        ckpt_files = list(Path(ckpt_dir).glob("epoch=*.pt"))
        ok_ckpt = len(ckpt_files) > 0

        ok = ok_model and ok_ckpt
        report(
            "TinySleepNet + time_domain trains 1 epoch",
            ok,
            f"model_returned={ok_model}, checkpoints={len(ckpt_files)}",
        )
        return ok, trained_model, ds


# ---------------------------------------------------------------------------
# Test 2: SeqSleepNet + time_frequency
# ---------------------------------------------------------------------------

def test_seqsleepnet_time_frequency():
    """Train SeqSleepNet for 1 epoch on fake data with time_frequency preset."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir, \
         tempfile.TemporaryDirectory() as ckpt_dir:
        data_dir = Path(data_dir)
        create_fake_data(data_dir, n_subjects=3, duration_sec=600.0)

        ds = MultiSubjectFakeEDFDataset(
            root=str(data_dir),
            subject_ids=["SUB00", "SUB01", "SUB02"],
            channels=["EEG", "EOG", "EMG"],
            pipelines="time_frequency",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        model = SeqSleepNet(n_classes=5, in_chan=3, F=129, D=32)

        train_loader, valid_loader = make_dataloaders(
            ds, batch_size=2, num_workers=0, fold=0,
        )

        print(f"\n  [info] SeqSleepNet: train_loader has {len(train_loader)} batches, "
              f"valid_loader has {len(valid_loader)} batches")

        trained_model = Trainer.train(
            model=model,
            dataset=(train_loader, valid_loader),
            checkpoint_path=ckpt_dir,
            max_epochs=1,
            lr=1e-5,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            gpu_id=None,
            seed=42,
        )

        ok_model = trained_model is not None
        ckpt_files = list(Path(ckpt_dir).glob("epoch=*.pt"))
        ok_ckpt = len(ckpt_files) > 0

        ok = ok_model and ok_ckpt
        report(
            "SeqSleepNet + time_frequency trains 1 epoch",
            ok,
            f"model_returned={ok_model}, checkpoints={len(ckpt_files)}",
        )
        return ok, trained_model, ds


# ---------------------------------------------------------------------------
# Test 3: Trainer.evaluate with dict batches
# ---------------------------------------------------------------------------

def test_evaluate_with_dict_batch():
    """Evaluate a trained model using dict-batch DataLoader.

    NOTE: Trainer.evaluate currently unpacks as (inputs, targets) = batch,
    which fails with dict batches. If the other agent has patched evaluate
    to support dict batches, this test will pass. Otherwise it will fail
    with a clear message indicating evaluate needs the same dict-batch
    support that _step already has.
    """
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir, \
         tempfile.TemporaryDirectory() as ckpt_dir:
        data_dir = Path(data_dir)
        create_fake_data(data_dir, n_subjects=2, duration_sec=300.0)

        ds = MultiSubjectFakeEDFDataset(
            root=str(data_dir),
            subject_ids=["SUB00", "SUB01"],
            channels=["EEG", "EOG", "EMG"],
            pipelines="time_domain",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        model = TinySleepNet(n_classes=5, in_chan=3, sf=100)

        # Build a simple eval DataLoader
        eval_indices = list(range(min(6, len(ds))))
        eval_subset = torch.utils.data.Subset(ds, eval_indices)
        eval_loader = DataLoader(
            eval_subset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=dict_collate_fn,
        )

        results = Trainer.evaluate(
            model=model,
            dataset=eval_loader,
            batch_size=2,
            num_workers=0,
            gpu_id=None,
            seed=42,
        )

        ok_dict = isinstance(results, dict)
        ok_keys = "accuracy" in results if ok_dict else False
        ok = ok_dict and ok_keys
        report(
            "Trainer.evaluate with dict batches",
            ok,
            f"results_type={type(results).__name__}, keys={list(results.keys()) if ok_dict else 'N/A'}",
        )
        return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

