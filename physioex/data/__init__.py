"""PhysioEx data module public API.

The canonical data layer is the raw-EDF, lazy-loading ``BasePhysioDataset``
stack exported below (resolve datasets by name via
``physioex.data.datasets.get_dataset``).

DEPRECATED: the legacy preprocessed-array layer (``PhysioExDataset`` /
``DataReader``) is kept only for backward compatibility and emits a
``DeprecationWarning`` on instantiation. It is not re-exported here (its
internal imports would create circular-import issues); import it explicitly:
``from physioex.data.dataset import PhysioExDataset``.
"""

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
