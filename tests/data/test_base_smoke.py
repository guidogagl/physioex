"""Smoke tests for BasePhysioDataset via the synthetic fixture.

Doubles as the Fase A scaffolding check (fixtures + markers + collection).
"""
import pytest
import torch


@pytest.mark.unit
def test_fake_dataset_len_and_item(fake_edf_dataset):
    ds = fake_edf_dataset(stages=["W", "N1", "N2", "N3", "R"], channels=["EEG"])
    assert len(ds) > 0
    item = ds[0]
    assert "signals" in item and "labels" in item and "channel_order" in item
    assert isinstance(item["labels"], torch.Tensor)


@pytest.mark.unit
def test_fake_dataset_split_shapes(fake_edf_dataset):
    ds = fake_edf_dataset(channels=["EEG", "EOG"])
    train_idx, valid, test = ds.split(fold=0)
    assert len(train_idx) + len(valid) + len(test) > 0


@pytest.mark.unit
def test_channel_count_reflected(fake_edf_dataset):
    ds = fake_edf_dataset(channels=["EEG", "EOG", "EMG"])
    item = ds[0]
    # channel_order lists one entry per requested channel
    assert len(item["channel_order"]) == 3
