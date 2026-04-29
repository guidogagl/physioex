"""
Integration tests for Phase D: dict-returning BasePhysioDataset with the Trainer.

Tests:
  a) Dict batch detection in _step: pass a dict batch, verify no crash, returns (loss, acc).
  b) Tuple batch still works in _step: verify the legacy path still works.
  c) build_dataloaders with BasePhysioDataset: create a tiny synthetic dataset,
     call Trainer.build_dataloaders, verify 3 DataLoaders with dict batches.
  d) _step with stacked-channel dict batch from collate: manually build a collated
     dict batch and run through _step.
  e) evaluate() with dict DataLoader: pass a DataLoader yielding dict batches.
  f) train() with BasePhysioDataset: run 1 epoch of training on synthetic data.

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_trainer_integration.py
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from physioex.train.trainer import Trainer

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

N_CLASSES = 5
SEQ_LEN = 5
SAMPLES_PER_EPOCH = 100  # simple raw signal: 100 samples per 30s epoch at ~3.33 Hz


class TinyModel(torch.nn.Module):
    """Accepts (B, L, C, T) and outputs (B, L, N_CLASSES)."""
    def __init__(self, n_features=SAMPLES_PER_EPOCH, n_classes=N_CLASSES):
        super().__init__()
        self.fc = torch.nn.Linear(n_features, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        # x shape: (B, L, C, T) or (B, L, features)
        b, seq = x.shape[0], x.shape[1]
        x = x.reshape(b * seq, -1)
        # Take first n_features to handle variable channel counts
        x = x[..., :self.fc.in_features]
        out = self.fc(x)
        return out.reshape(b, seq, self.n_classes)


def make_dict_batch(batch_size=2, seq_len=SEQ_LEN, n_channels=3, samples=SAMPLES_PER_EPOCH):
    """Create a dict batch as dict_collate_fn would produce."""
    channel_names = ["C4-M2", "EOG", "EMG"][:n_channels]
    signals = {
        name: torch.randn(batch_size, seq_len, samples)
        for name in channel_names
    }
    labels = torch.randint(0, N_CLASSES, (batch_size, seq_len))
    return {
        "signals": signals,
        "channel_order": channel_names,
        "labels": labels,
    }


def make_tuple_batch(batch_size=2, seq_len=SEQ_LEN, n_channels=3, samples=SAMPLES_PER_EPOCH):
    """Create a legacy tuple batch (inputs, targets)."""
    inputs = torch.randn(batch_size, seq_len, n_channels, samples)
    targets = torch.randint(0, N_CLASSES, (batch_size, seq_len))
    return inputs, targets


# -------------------------------------------------------------------
# Test a: Dict batch detection in _step
# -------------------------------------------------------------------
def test_step_dict_batch():
    try:
        model = TinyModel(n_features=SAMPLES_PER_EPOCH * 3)  # 3 channels stacked
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        device = torch.device("cpu")
        batch = make_dict_batch(batch_size=2, n_channels=3)

        result = Trainer._step(model, batch, loss_fn, device)
        loss, acc = result
        assert isinstance(loss, torch.Tensor), f"loss should be Tensor, got {type(loss)}"
        assert isinstance(acc, float), f"acc should be float, got {type(acc)}"
        assert 0.0 <= acc <= 1.0, f"acc={acc} out of [0,1]"
        report("a: _step with dict batch", True)
    except Exception as exc:
        report("a: _step with dict batch", False, str(exc))


# -------------------------------------------------------------------
# Test b: Tuple batch still works in _step
# -------------------------------------------------------------------
def test_step_tuple_batch():
    try:
        model = TinyModel(n_features=SAMPLES_PER_EPOCH * 3)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        device = torch.device("cpu")
        batch = make_tuple_batch(batch_size=2, n_channels=3)

        result = Trainer._step(model, batch, loss_fn, device)
        loss, acc = result
        assert isinstance(loss, torch.Tensor), f"loss should be Tensor, got {type(loss)}"
        assert isinstance(acc, float), f"acc should be float, got {type(acc)}"
        assert 0.0 <= acc <= 1.0, f"acc={acc} out of [0,1]"
        report("b: _step with tuple batch (legacy)", True)
    except Exception as exc:
        report("b: _step with tuple batch (legacy)", False, str(exc))


# -------------------------------------------------------------------
# Test c: build_dataloaders with BasePhysioDataset
# -------------------------------------------------------------------
def test_build_dataloaders_base_dataset():
    try:
        import pyedflib
        from physioex.data.base import BasePhysioDataset, SubjectSpec

        # Reuse write_fake_edf from test_raw_dataset_integration.py pattern
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            data_dir = Path(data_dir)
            _write_multi_subject_data(data_dir, n_subjects=10, duration_sec=300)

            ds = _MultiSubjectFakeDataset(
                root=str(data_dir),
                n_subjects=10,
                channels=["EEG", "EOG"],
                pipelines="raw",
                sequence_length=SEQ_LEN,
                cache_dir=cache_dir,
            )

            train_loader, valid_loader, test_loader = Trainer.build_dataloaders(
                dataset=ds,
                train_batch_size=2,
                eval_batch_size=1,
                num_workers=0,
                fold=0,
            )

            assert isinstance(train_loader, DataLoader)
            assert isinstance(valid_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)

            # Iterate train loader and verify dict batches
            batch = next(iter(train_loader))
            assert isinstance(batch, dict), f"Expected dict batch, got {type(batch)}"
            assert "signals" in batch, "Missing 'signals' key in batch"
            assert "labels" in batch, "Missing 'labels' key in batch"
            assert "channel_order" in batch, "Missing 'channel_order' key in batch"

            # Check shapes: signals[name] should be (B, L, T)
            for name in batch["channel_order"]:
                sig = batch["signals"][name]
                assert sig.ndim == 3, f"signal {name} should be 3D (B,L,T), got {sig.ndim}D"
                assert sig.shape[1] == SEQ_LEN, f"signal seq_len={sig.shape[1]}, expected {SEQ_LEN}"

            assert batch["labels"].ndim == 2, f"labels should be 2D (B,L), got {batch['labels'].ndim}D"

            report("c: build_dataloaders with BasePhysioDataset", True)
    except Exception as exc:
        report("c: build_dataloaders with BasePhysioDataset", False, str(exc))


# -------------------------------------------------------------------
# Test d: _step with collated dict batch (stack_channels)
# -------------------------------------------------------------------
def test_step_with_collated_dict():
    try:
        from physioex.data.collate import stack_channels, dict_collate_fn

        # Build individual items (as BasePhysioDataset would return)
        items = []
        for _ in range(3):
            item = {
                "signals": {
                    "C4-M2": torch.randn(SEQ_LEN, SAMPLES_PER_EPOCH),
                    "EOG": torch.randn(SEQ_LEN, SAMPLES_PER_EPOCH),
                },
                "channel_order": ["C4-M2", "EOG"],
                "labels": torch.randint(0, N_CLASSES, (SEQ_LEN,)),
            }
            items.append(item)

        batch = dict_collate_fn(items)

        # Verify stack_channels produces correct tensor
        stacked = stack_channels(batch)
        assert stacked.shape == (3, SEQ_LEN, 2, SAMPLES_PER_EPOCH), f"Unexpected shape: {stacked.shape}"

        # Run through _step
        model = TinyModel(n_features=SAMPLES_PER_EPOCH * 2)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss, acc = Trainer._step(model, batch, loss_fn, torch.device("cpu"))
        assert isinstance(loss, torch.Tensor)
        report("d: _step with collated dict batch", True)
    except Exception as exc:
        report("d: _step with collated dict batch", False, str(exc))


# -------------------------------------------------------------------
# Test e: evaluate() with dict DataLoader
# -------------------------------------------------------------------
def test_evaluate_dict_loader():
    try:
        from physioex.data.collate import dict_collate_fn
        from torch.utils.data import Dataset

        class FakeDictDataset(Dataset):
            def __init__(self, n=10):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {
                    "signals": {
                        "C4-M2": torch.randn(SEQ_LEN, SAMPLES_PER_EPOCH),
                        "EOG": torch.randn(SEQ_LEN, SAMPLES_PER_EPOCH),
                    },
                    "channel_order": ["C4-M2", "EOG"],
                    "labels": torch.randint(0, N_CLASSES, (SEQ_LEN,)),
                }

        loader = DataLoader(
            FakeDictDataset(10), batch_size=2, collate_fn=dict_collate_fn
        )

        model = TinyModel(n_features=SAMPLES_PER_EPOCH * 2)

        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            results = Trainer.evaluate(
                model=model,
                dataset=loader,
                gpu_id=None,
            )
        os.chdir(orig_cwd)

        assert "accuracy" in results, f"Missing 'accuracy' key; keys={list(results.keys())}"
        assert 0.0 <= results["accuracy"] <= 1.0
        report("e: evaluate() with dict DataLoader", True)
    except Exception as exc:
        report("e: evaluate() with dict DataLoader", False, str(exc))


# -------------------------------------------------------------------
# Test f: train() with BasePhysioDataset (1 epoch)
# -------------------------------------------------------------------
def test_train_base_dataset():
    try:
        import pyedflib
        from physioex.data.base import BasePhysioDataset, SubjectSpec

        with tempfile.TemporaryDirectory() as data_dir, \
             tempfile.TemporaryDirectory() as cache_dir, \
             tempfile.TemporaryDirectory() as ckpt_dir:
            data_dir = Path(data_dir)
            _write_multi_subject_data(data_dir, n_subjects=10, duration_sec=300)

            ds = _MultiSubjectFakeDataset(
                root=str(data_dir),
                n_subjects=10,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=SEQ_LEN,
                cache_dir=cache_dir,
            )

            model = TinyModel(n_features=SAMPLES_PER_EPOCH)

            orig_cwd = os.getcwd()
            os.chdir(ckpt_dir)

            model = Trainer.train(
                model=model,
                dataset=ds,
                max_epochs=1,
                train_batch_size=2,
                eval_batch_size=1,
                num_workers=0,
                checkpoint_path=os.path.join(ckpt_dir, "ckpts"),
                gpu_id=None,
            )
            os.chdir(orig_cwd)

            # Model should still be functional after training
            x = torch.randn(1, SEQ_LEN, 1, SAMPLES_PER_EPOCH)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, SEQ_LEN, N_CLASSES), f"Output shape mismatch: {out.shape}"
            report("f: train() with BasePhysioDataset (1 epoch)", True)
    except Exception as exc:
        report("f: train() with BasePhysioDataset (1 epoch)", False, str(exc))


# -------------------------------------------------------------------
# Shared helpers: multi-subject fake dataset
# -------------------------------------------------------------------

def _write_multi_subject_data(data_dir: Path, n_subjects: int = 4, duration_sec: float = 300.0):
    """Write multiple fake EDF subjects into data_dir."""
    import pyedflib

    n_epochs = int(duration_sec / 30.0)
    stages = ["W", "N1", "N2", "N3", "R"]

    for i in range(n_subjects):
        sid = f"SUB{i:02d}"
        rng = np.random.default_rng(seed=i)

        # Write signal EDF
        edf_path = data_dir / f"{sid}.edf"
        channel_names = ["C4-M2", "C3-M1", "EOG", "EMG"]
        n_channels = len(channel_names)
        fs = 100
        signals = []
        headers = []
        for ch in range(n_channels):
            x = rng.standard_normal(int(duration_sec * fs)).astype(np.float64) * 20.0
            signals.append(x)
            headers.append({
                "label": channel_names[ch],
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": -300.0, "physical_max": 300.0,
                "digital_min": -32768, "digital_max": 32767,
                "transducer": "", "prefilter": "",
            })
        writer = pyedflib.EdfWriter(str(edf_path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            writer.setSignalHeaders(headers)
            writer.writeSamples(signals)
        finally:
            writer.close()

        # Write label EDF
        label_path = data_dir / f"{sid}_sleepscoring.edf"
        stage_list = [stages[j % len(stages)] for j in range(n_epochs)]
        writer = pyedflib.EdfWriter(str(label_path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            writer.setSignalHeaders([{
                "label": "dummy", "dimension": "uV", "sample_frequency": 1,
                "physical_min": -1.0, "physical_max": 1.0,
                "digital_min": -32768, "digital_max": 32767,
                "transducer": "", "prefilter": "",
            }])
            writer.writeSamples([np.zeros(int(duration_sec), dtype=np.float64)])
            for j, s in enumerate(stage_list):
                writer.writeAnnotation(j * 30.0, 30.0, s)
        finally:
            writer.close()


class _MultiSubjectFakeDataset:
    """Thin wrapper that creates a FakeEDFDataset-like subclass with multiple subjects."""

    def __new__(cls, root, n_subjects, **kwargs):
        from physioex.data.base import BasePhysioDataset, SubjectSpec

        class _Inner(BasePhysioDataset):
            DATASET_NAME = "fake_trainer_integration"
            DEFAULT_EPOCH_LENGTH_SEC = 30.0
            CHANNEL_PREFERENCES = {
                "EEG": [("C4", "M2"), "C4-M2", "C3-M1", "EEG"],
                "EOG": ["EOG"],
                "EMG": ["EMG"],
            }

            def __init__(self, root, _n_subjects, **kw):
                self._n_subs = _n_subjects
                super().__init__(root=root, **kw)

            def _list_subjects(self):
                root = Path(self.root)
                specs = []
                for i in range(self._n_subs):
                    sid = f"SUB{i:02d}"
                    specs.append(SubjectSpec(
                        subject_id=sid,
                        edf_path=root / f"{sid}.edf",
                        label_path=root / f"{sid}_sleepscoring.edf",
                    ))
                return specs

            def _read_subject_labels(self, spec):
                import pyedflib
                with pyedflib.EdfReader(str(spec.label_path)) as f:
                    onsets, durations, stage_strs = f.readAnnotations()
                stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
                if len(stage_strs) == 0:
                    return np.array([], dtype=np.int16)
                total = max(float(o) + float(d) for o, d in zip(onsets, durations))
                n_epochs = int(total // self.epoch_length_sec)
                labels = np.full(n_epochs, -1, dtype=np.int16)
                for onset, dur, s in zip(onsets, durations, stage_strs):
                    i0 = int(round(float(onset) / self.epoch_length_sec))
                    i1 = int(round((float(onset) + float(dur)) / self.epoch_length_sec))
                    if i1 > n_epochs:
                        i1 = n_epochs
                    labels[i0:i1] = stage_map.get(str(s).strip(), -1)
                return labels

        return _Inner(root=root, _n_subjects=n_subjects, **kwargs)


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Trainer integration tests (Phase D: dict batch support)")
    print("=" * 60)

    test_step_dict_batch()
    test_step_tuple_batch()
    test_build_dataloaders_base_dataset()
    test_step_with_collated_dict()
    test_evaluate_dict_loader()
    test_train_base_dataset()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
