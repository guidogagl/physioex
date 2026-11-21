"""Whole-night replication support: paper metrics, night iteration, alignment,
kappa evaluation, and the models' ``predict_night`` interface.

All tests are synthetic (randomly initialised models, fake datasets,
``tmp_path`` JSON files); no network or checkpoint downloads.
"""
import datetime
import json

import numpy as np
import pytest
import torch

from physioex.train.replication import (
    PaperTarget,
    align,
    check,
    evaluate,
    format_table,
    iter_nights,
    load_paper_metrics,
)
from physioex.train.metrics import median_record_kappa, pooled_kappa


# ── load_paper_metrics ───────────────────────────────────────────────


def test_load_paper_metrics_defaults_and_overrides(tmp_path):
    doc = {
        "model": "wav2sleep",
        "source": "Carter & Tarassenko 2024",
        "metric": "pooled_kappa",
        "tolerance": 0.10,
        "targets": [
            {"dataset": "shhs", "condition": "test", "value": 0.72},
            {
                "dataset": "mros",
                "condition": "test",
                "value": 0.70,
                "metric": "median_record_kappa",
                "tolerance": 0.05,
                "assert": False,
                "requires": ["mros", "mesa"],
                "note": "census split",
            },
        ],
    }
    p = tmp_path / "paper.json"
    p.write_text(json.dumps(doc))
    header, targets = load_paper_metrics(p)

    assert header == {
        "model": "wav2sleep",
        "source": "Carter & Tarassenko 2024",
        "metric": "pooled_kappa",
        "tolerance": 0.10,
    }
    assert len(targets) == 2
    t0, t1 = targets
    # defaults from the top level
    assert (t0.dataset, t0.condition, t0.value) == ("shhs", "test", 0.72)
    assert t0.metric == "pooled_kappa"
    assert t0.tolerance == 0.10
    assert t0.assert_ is True
    assert t0.requires == ["shhs"]
    assert t0.note == ""
    # per-target overrides
    assert t1.metric == "median_record_kappa"
    assert t1.tolerance == 0.05
    assert t1.assert_ is False
    assert t1.requires == ["mros", "mesa"]
    assert t1.note == "census split"


def test_load_paper_metrics_rejects_unknown_metric(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "model": "x",
                "source": "y",
                "metric": "bad_metric",
                "tolerance": 0.1,
                "targets": [{"dataset": "a", "condition": "test", "value": 1.0}],
            }
        )
    )
    with pytest.raises(ValueError, match="metric"):
        load_paper_metrics(p)


# ── check ────────────────────────────────────────────────────────────


def _target(dataset="shhs", **kw):
    base = dict(dataset=dataset, condition="test", value=0.72)
    base.update(kw)
    return PaperTarget(**base)


def test_check_pass_fail_report_skip():
    results = {
        ("shhs", "test"): {"pooled_kappa": 0.75, "median_record_kappa": 0.70},
        ("mros", "test"): {"pooled_kappa": 0.60, "median_record_kappa": 0.61},
        ("mesa", "test"): {"pooled_kappa": 0.50, "median_record_kappa": 0.51},
    }
    targets = [
        _target(),  # |0.75 - 0.72| = 0.03 <= 0.05 -> PASS
        _target(dataset="mros"),  # |0.60 - 0.72| > 0.05 -> FAIL
        _target(dataset="shhs", assert_=False),  # REPORT
        _target(dataset="shhs", requires=["shhs", "nonexistent"]),  # SKIP
    ]
    rows = check(results, targets)
    assert [r["status"] for r in rows] == ["PASS", "FAIL", "REPORT", "SKIP"]

    r_pass, r_fail, r_report, r_skip = rows
    assert r_pass["paper"] == 0.72
    assert r_pass["ours"] == pytest.approx(0.75)
    assert r_pass["delta"] == pytest.approx(0.03)
    assert r_pass["tolerance"] == 0.05
    assert r_fail["delta"] == pytest.approx(0.60 - 0.72)
    assert r_report["ours"] == pytest.approx(0.75)
    assert r_skip["ours"] is None and r_skip["delta"] is None
    assert "nonexistent" in r_skip["note"]


def test_check_skip_reports_all_missing_datasets():
    targets = [_target(requires=["shhs", "mros", "mesa"])]
    rows = check({}, targets)
    assert rows[0]["status"] == "SKIP"
    assert "shhs" in rows[0]["note"] and "mros" in rows[0]["note"]


def test_format_table_shapes():
    rows = [
        {
            "dataset": "shhs",
            "condition": "test",
            "metric": "pooled_kappa",
            "paper": 0.72,
            "ours": 0.751234,
            "delta": 0.03,
            "tolerance": 0.05,
            "status": "PASS",
            "note": "",
        },
        {
            "dataset": "mros",
            "condition": "test",
            "metric": "pooled_kappa",
            "paper": 0.70,
            "ours": None,
            "delta": None,
            "tolerance": 0.05,
            "status": "SKIP",
            "note": "missing results for: mros",
        },
    ]
    table = format_table(rows)
    lines = table.split("\n")
    assert len(lines) == 4  # header, separator, 2 rows
    assert lines[0].startswith("dataset") and "status" in lines[0]
    # the status column starts where the header says so; both rows align there
    status_col = lines[0].index("status")
    assert lines[2][status_col : status_col + 4] == "PASS"
    assert lines[3][status_col : status_col + 4] == "SKIP"
    assert lines[3].endswith("missing results for: mros")


# ── align ────────────────────────────────────────────────────────────


def test_align_predictions_longer_than_labels():
    pred = np.arange(1200)
    labels = np.arange(1150) % 4
    p, l, dropped = align(pred, labels)
    assert p.shape == (1150,) and l.shape == (1150,)
    assert dropped == 0
    np.testing.assert_array_equal(p, pred[:1150])
    np.testing.assert_array_equal(l, labels)


def test_align_labels_longer_than_predictions():
    pred = np.arange(1200)
    labels = np.arange(1250) % 4
    p, l, dropped = align(pred, labels)
    assert p.shape == (1200,) and l.shape == (1200,)
    assert dropped == 50  # the 50 labelled epochs beyond the grid
    np.testing.assert_array_equal(l, labels[:1200])


def test_align_equal_lengths():
    p, l, dropped = align(np.arange(10), np.arange(10) % 4)
    assert p.shape == (10,) and l.shape == (10,) and dropped == 0


# ── evaluate ─────────────────────────────────────────────────────────


def test_evaluate_matches_metrics_module():
    rng = np.random.default_rng(0)
    trues, preds = [], []
    for n in (20, 30, 40):
        t = rng.integers(0, 4, n)
        p = np.where(rng.random(n) < 0.7, t, rng.integers(0, 4, n))
        t[:3] = -1
        trues.append(t)
        preds.append(p)
    out = evaluate(trues, preds, 4)
    assert out["pooled_kappa"] == pytest.approx(pooled_kappa(trues, preds, 4))
    assert out["median_record_kappa"] == pytest.approx(median_record_kappa(trues, preds, 4))
    assert out["n_records"] == 3
    assert out["n_epochs"] == int(sum((t != -1).sum() for t in trues))
    assert isinstance(out["per_record_kappa"], list) and len(out["per_record_kappa"]) == 3
    assert not np.isnan(out["pooled_kappa"])


def test_evaluate_mismatched_record_counts():
    with pytest.raises(ValueError, match="records"):
        evaluate([np.arange(4)], [np.arange(4), np.arange(4)], 4)


# ── iter_nights (tiny in-memory fake dataset) ────────────────────────


class _FakeSpec:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.edf_path = None
        self.label_path = None
        self.external_meta = {}


class _FakeRecordingDataset:
    """Minimal stand-in for a whole-night ``BasePhysioDataset``.

    Only the attributes ``iter_nights`` reads: ``sequence_length``,
    ``DATASET_NAME``, ``_subjects`` and item access.
    """

    DATASET_NAME = "fake"

    def __init__(self, n_subjects=2, n_epochs=50, samples_per_epoch=8, has_metadata=True):
        self._has_metadata = has_metadata
        rng = np.random.default_rng(0)
        self.sequence_length = 0
        self._subjects = [_FakeSpec(f"sub{i:03d}") for i in range(n_subjects)]
        self._data = {
            s.subject_id: {
                "signals": {
                    "ECG_0": rng.standard_normal((n_epochs, samples_per_epoch)).astype(np.float32),
                    "RESP_0": rng.standard_normal((n_epochs, samples_per_epoch)).astype(np.float32),
                },
                "labels": rng.integers(0, 4, n_epochs).astype(np.int64),
            }
            for s in self._subjects
        }
        self._metadata = {
            s.subject_id: {
                "age": 30.0 + i,
                "sex": "M" if i % 2 else "F",
                "recording_start": datetime.time(22, 0),
            }
            for i, s in enumerate(self._subjects)
        }

    def __getitem__(self, idx):
        sid = self._subjects[idx].subject_id
        return {
            "signals": {k: torch.from_numpy(v.copy()) for k, v in self._data[sid]["signals"].items()},
            "labels": torch.from_numpy(self._data[sid]["labels"].copy()),
        }

    def subject_metadata(self, subject_id):
        if not self._has_metadata:
            raise AttributeError("subject_metadata")
        return dict(self._metadata[subject_id])


def test_iter_nights_fake_dataset():
    ds = _FakeRecordingDataset(n_subjects=2)
    nights = list(iter_nights(ds, ["sub001", "sub000"]))
    assert [sid for sid, *_ in nights] == ["sub001", "sub000"]
    for sid, signals, labels, meta in nights:
        assert set(signals) == {"ECG_0", "RESP_0"}
        for name, arr in signals.items():
            assert arr.dtype == np.float32 and arr.shape == (50, 8)
        assert labels.dtype == np.int64 and labels.shape == (50,)
        assert meta == ds.subject_metadata(sid)
    # item 0 must be subject sub000, item 1 sub001 (recording mode)
    sid, signals, labels, _ = nights[1]
    assert sid == "sub000"
    np.testing.assert_array_equal(labels, ds._data["sub000"]["labels"])


def test_iter_nights_dataset_without_metadata():
    ds = _FakeRecordingDataset(n_subjects=1, has_metadata=False)
    nights = list(iter_nights(ds, ["sub000"]))
    assert nights[0][3] == {}


def test_iter_nights_rejects_sequence_mode():
    ds = _FakeRecordingDataset(n_subjects=1)
    ds.sequence_length = 21
    with pytest.raises(ValueError, match="whole-night"):
        list(iter_nights(ds, ["sub000"]))


def test_iter_nights_unknown_subject():
    ds = _FakeRecordingDataset(n_subjects=1)
    with pytest.raises(KeyError, match="not in dataset"):
        list(iter_nights(ds, ["nobody"]))


# ── predict_night on the three models ─────────────────────────────────


def _wav2sleep_model():
    from physioex.models.wav2sleep import (
        MultiModalAttentionEmbedder,
        SequenceCNN,
        SignalEncoders,
        Wav2Sleep,
    )

    torch.manual_seed(0)
    model = Wav2Sleep(
        signal_encoders=SignalEncoders(
            signal_map={"ECG": "ecg", "PPG": "ppg", "ABD": "abd", "THX": "thx"},
            feature_dim=16,
            activation="gelu",
            norm="instance",
        ),
        epoch_mixer=MultiModalAttentionEmbedder(16, layers=1, dim_ff=32, nhead=4),
        sequence_mixer=SequenceCNN(feature_dim=16, num_layers=1),
        num_classes=4,
    ).eval()
    return model


def test_wav2sleep_predict_night_ecg_only():
    from physioex.models.wav2sleep import SAMPLES_PER_EPOCH

    model = _wav2sleep_model()
    L = 40
    signals = {"ECG": np.random.default_rng(1).standard_normal((L, SAMPLES_PER_EPOCH["ECG"])).astype(np.float32)}
    preds = model.predict_night(signals)
    assert preds.shape == (L,)
    assert preds.dtype == np.int64
    assert preds.min() >= 0 and preds.max() < 4
    # unknown keys are ignored
    signals["XX"] = np.zeros((L, 4))
    np.testing.assert_array_equal(model.predict_night(signals), preds)


def test_wav2sleep_predict_night_full_grid():
    from physioex.models.wav2sleep import SAMPLES_PER_EPOCH

    model = _wav2sleep_model()
    L = 1200
    signals = {
        sig: np.random.default_rng(2).standard_normal((L, SAMPLES_PER_EPOCH[sig])).astype(np.float32)
        for sig in ("ECG", "PPG", "ABD", "THX")
    }
    preds = model.predict_night(signals)
    assert preds.shape == (L,) and preds.dtype == np.int64
    assert preds.min() >= 0 and preds.max() < 4


def _sleepppgnet_model():
    from physioex.models.sleepppgnet import SAMPLES_PER_EPOCH, SleepPPGNet

    torch.manual_seed(0)
    # Same architecture, fewer epochs: fast to build and to run.
    return SleepPPGNet(n_epochs=32, samples_per_epoch=SAMPLES_PER_EPOCH).eval()


def test_sleepppgnet_predict_night():
    from physioex.models.sleepppgnet import SAMPLES_PER_EPOCH

    model = _sleepppgnet_model()
    L = 20
    signals = {"PPG": np.random.default_rng(3).standard_normal((L, SAMPLES_PER_EPOCH)).astype(np.float32)}
    preds = model.predict_night(signals)
    assert preds.shape == (L,)
    assert preds.dtype == np.int64
    assert preds.min() >= 0 and preds.max() < 4
    with pytest.raises(ValueError, match="PPG"):
        model.predict_night({"ECG": signals["PPG"]})


def _jones_model(variant="primary"):
    from physioex.models.jones_ecg import JonesECGNet

    torch.manual_seed(0)
    return JonesECGNet(variant=variant).eval()


def test_jones_predict_night():
    from physioex.models.jones_ecg import JonesECGNet

    model = _jones_model()
    L = 30
    signals = {"ECG": np.clip(np.random.default_rng(4).standard_normal((L, 7680)).astype(np.float32) * 0.2, -1, 1)}
    metadata = {
        "age": 45.0,
        "sex": "M",
        "recording_start": datetime.time(23, 30),
    }
    preds = model.predict_night(signals, metadata)
    assert preds.shape == (L,)
    assert preds.dtype == np.int64
    assert preds.min() >= 0 and preds.max() < 5
    # determinism: no random state is consumed by the call itself
    np.testing.assert_array_equal(model.predict_night(signals, metadata), preds)
    # midpoint of the night: build_additional agrees
    expected_offset = JonesECGNet.midnight_offset(datetime.time(23, 30))
    assert metadata["recording_start"] is not None
    assert expected_offset < 0
    with pytest.raises(ValueError, match="ECG"):
        model.predict_night({"PPG": signals["ECG"]}, metadata)


def test_jones_predict_night_missing_metadata(caplog):
    """age/sex/recording_start all missing: variant 'primary' falls back to its
    noise-filled columns and warns once per missing key."""
    import logging

    model = _jones_model("primary")
    L = 16
    signals = {"ECG": np.random.default_rng(5).standard_normal((L, 7680)).astype(np.float32)}
    with caplog.at_level(logging.WARNING, logger="physioex.models.jones_ecg"):
        preds = model.predict_night(signals)
    assert preds.shape == (L,) and preds.dtype == np.int64
    assert preds.min() >= 0 and preds.max() < 5
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    for key in ("'age'", "'sex'", "'recording_start'"):
        assert any(key in w for w in warnings)


def test_predict_night_restores_training_mode():
    model = _jones_model("primary")
    model.train()
    L = 8
    signals = {"ECG": np.random.default_rng(6).standard_normal((L, 7680)).astype(np.float32)}
    model.predict_night(signals)
    assert model.training is True
