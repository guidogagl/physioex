"""Regression tests for trainer fixes.

- A1: ``_get_parameters_from_config`` robust to missing / partial YAML.
- B7: ``Trainer.voting_evaluate`` sliding-window output shape.

(The former D6 source-grep tests, which asserted the ``_train_step`` /
``_eval_step`` tuple arity by scraping the source, were removed: that arity is
now exercised behaviorally end-to-end by ``tests/test_trainer_integration.py``.)
"""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from physioex.train.trainer import Trainer, _get_parameters_from_config


def test_config_missing_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _get_parameters_from_config()
    assert isinstance(result, dict)
    assert all(v is None for v in result.values())


def test_config_reads_yaml(tmp_path, monkeypatch):
    (tmp_path / "PHYSIOEX_CONFIG.yaml").write_text(
        "Trainer:\n  max_epochs: 42\n  train_batch_size: 64\n"
    )
    monkeypatch.chdir(tmp_path)
    result = _get_parameters_from_config()
    assert result["max_epochs"] == 42
    assert result["train_batch_size"] == 64


def test_config_no_trainer_section(tmp_path, monkeypatch):
    (tmp_path / "PHYSIOEX_CONFIG.yaml").write_text("PhysioExDataset:\n  datasets:\n    - hmc\n")
    monkeypatch.chdir(tmp_path)
    result = _get_parameters_from_config()
    assert isinstance(result, dict)
    assert all(v is None for v in result.values())


def test_voting_evaluate(tmp_path, monkeypatch):
    N_CLASSES, FEATURES, NIGHT_LEN, L = 5, 16, 50, 10

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(FEATURES, N_CLASSES)

        def forward(self, x):
            b, seq = x.shape[0], x.shape[1]
            return self.fc(x.reshape(b * seq, -1)).reshape(b, seq, N_CLASSES)

    class FakeNightDataset(Dataset):
        def __init__(self, n_subjects=3):
            self.n = n_subjects

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return torch.randn(NIGHT_LEN, FEATURES), torch.randint(0, N_CLASSES, (NIGHT_LEN,))

    loader = DataLoader(FakeNightDataset(3), batch_size=1, shuffle=False)
    # Isolate from any real PHYSIOEX_CONFIG.yaml in the cwd.
    monkeypatch.chdir(tmp_path)
    results = Trainer.voting_evaluate(model=TinyModel(), dataset=loader, L=L, gpu_id=None)

    expected = {
        "accuracy", "f1_score", "precision", "recall",
        "cohen_kappa", "confusion_matrix", "support",
    }
    assert set(results.keys()) == expected
    acc = results["accuracy"]
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0
