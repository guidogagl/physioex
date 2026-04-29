"""PhysioEx data module public API."""
# Legacy API (preserved for backward compat)
# -- the legacy physioex/data/dataset.py file uses internal imports
# (from datareader import DataReader) so we do NOT import from
# dataset/datareader here to avoid circular-import issues.
# Users who want the legacy PhysioExDataset can still do:
#   from physioex.data.dataset import PhysioExDataset

# New API
from physioex.data.pipeline import (
    PreprocessingStep,
    PreprocessingPipeline,
    CompiledPipeline,
    CompiledStep,
)
from physioex.data.steps import (
    Identity,
    BandpassFilter,
    NotchFilter,
    Resample,
    ZScoreNormalize,
    XSleepNetSpectrogram,
)
from physioex.data.presets import get_preset, available_presets, PRESETS
from physioex.data.base import BasePhysioDataset, SubjectSpec
from physioex.data.multi import MultiDataset
from physioex.data.cache import ChannelCache, SCHEMA_VERSION
from physioex.data.collate import dict_collate_fn, stack_channels, is_dict_batch
from physioex.data.readers import (
    EDFHeader,
    ResolvedChannel,
    ChannelNotAvailableError,
    DEFAULT_PREFERENCES,
    probe_edf_header,
    resolve_channels,
)
from physioex.data.events import (
    SleepEvent,
    map_events_to_epochs,
    events_to_dicts,
    dicts_to_events,
)

__all__ = [
    # Pipeline
    "PreprocessingStep",
    "PreprocessingPipeline",
    "CompiledPipeline",
    "CompiledStep",
    # Steps
    "Identity",
    "BandpassFilter",
    "NotchFilter",
    "Resample",
    "ZScoreNormalize",
    "XSleepNetSpectrogram",
    # Presets
    "get_preset",
    "available_presets",
    "PRESETS",
    # Dataset base
    "BasePhysioDataset",
    "SubjectSpec",
    "MultiDataset",
    # Cache
    "ChannelCache",
    "SCHEMA_VERSION",
    # Collate
    "dict_collate_fn",
    "stack_channels",
    "is_dict_batch",
    # Readers
    "EDFHeader",
    "ResolvedChannel",
    "ChannelNotAvailableError",
    "DEFAULT_PREFERENCES",
    "probe_edf_header",
    "resolve_channels",
    # Events
    "SleepEvent",
    "map_events_to_epochs",
    "events_to_dicts",
    "dicts_to_events",
]
