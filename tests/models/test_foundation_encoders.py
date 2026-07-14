"""Contract tests for the 9 foundation encoders.

Two tiers:
  - **Cheap (default CI)**: class attributes, inheritance, and that each
    model's PIPELINE_PRESET resolves to a real preprocessing pipeline. These
    never touch HuggingFace or heavy deps.
  - **Gated (@hf)**: actually build the encoder with random init and check the
    (B,L,C,T)->(B,L,D) contract. Deselected unless PHYSIOEX_TEST_HF=1, since
    building needs braindecode/transformers/checkpoints.
"""
import pytest
import torch

from physioex.models import (
    FoundationEncoder,
    CBraModEncoder,
    BENDREncoder,
    LaBraMEncoder,
    BIOTEncoder,
    SleepFMEncoder,
    TFCEncoder,
    REVEEncoder,
    SJEEncoder,
    NeuroLMEncoder,
)
from physioex.data.pipeline import PreprocessingPipeline, CompiledPipeline

VALID_STRATEGIES = {"first", "pad_fixed", "layout", "all"}

ENCODERS = [
    CBraModEncoder,
    BENDREncoder,
    LaBraMEncoder,
    BIOTEncoder,
    SleepFMEncoder,
    TFCEncoder,
    REVEEncoder,
    SJEEncoder,
    NeuroLMEncoder,
]


@pytest.mark.parametrize("cls", ENCODERS, ids=lambda c: c.MODEL_NAME)
def test_encoder_is_foundation_subclass(cls):
    assert issubclass(cls, FoundationEncoder)


@pytest.mark.parametrize("cls", ENCODERS, ids=lambda c: c.MODEL_NAME)
def test_encoder_class_attributes(cls):
    assert isinstance(cls.MODEL_NAME, str) and cls.MODEL_NAME != "foundation"
    assert isinstance(cls.PIPELINE_PRESET, str) and cls.PIPELINE_PRESET
    assert cls.CHANNEL_STRATEGY in VALID_STRATEGIES


@pytest.mark.parametrize("cls", ENCODERS, ids=lambda c: c.MODEL_NAME)
def test_encoder_get_pipeline_resolves(cls):
    """PIPELINE_PRESET must resolve to a real (compiled) pipeline."""
    pipe = cls.get_pipeline()
    assert isinstance(pipe, (PreprocessingPipeline, CompiledPipeline))
    # pipelines expose a stable content hash used for cache sharing
    assert isinstance(pipe.hash(), str) and pipe.hash()


def test_all_model_names_unique():
    names = [c.MODEL_NAME for c in ENCODERS]
    assert len(names) == len(set(names))


# ── Gated real-build tests ───────────────────────────────────────────

@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.parametrize("cls", ENCODERS, ids=lambda c: c.MODEL_NAME)
def test_encoder_encode_shape_random_init(cls):
    """Build with random init (no checkpoint) and check the embedding shape."""
    model = cls(in_chan=2, checkpoint_path=None)
    x = torch.randn(1, 3, 2, 3000)  # (B, L, C, T)
    out = model.encode(x)
    assert out.ndim == 3
    assert out.shape[0] == 1 and out.shape[1] == 3
    assert out.shape[2] == model.embedding_dim
