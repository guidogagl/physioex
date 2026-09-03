"""Public API for data readers."""
from physioex.data.readers.edf import (
    EDFHeader,
    ResolvedChannel,
    ChannelNotAvailableError,
    DEFAULT_PREFERENCES,
    KNOWN_MODALITIES,
    probe_edf_header,
    resolve_channels,
    read_channel,
    read_channels_from_edf,
)
from physioex.data.readers.vital import (
    VitalTrackNotFoundError,
    probe_vital_header,
    read_vital_channel,
    list_vital_tracks,
)
from physioex.data.readers.annotations import (
    NSRR_STAGE_MAP,
    parse_nsrr_xml,
    parse_tsv_annotations,
    parse_nsrr_xml_events,
    STAGES_STAGE_MAP,
    parse_stages_csv,
)

__all__ = [
    "EDFHeader",
    "ResolvedChannel",
    "ChannelNotAvailableError",
    "DEFAULT_PREFERENCES",
    "KNOWN_MODALITIES",
    "probe_edf_header",
    "resolve_channels",
    "read_channel",
    "read_channels_from_edf",
    "VitalTrackNotFoundError",
    "probe_vital_header",
    "read_vital_channel",
    "list_vital_tracks",
    "NSRR_STAGE_MAP",
    "parse_nsrr_xml",
    "parse_tsv_annotations",
    "parse_nsrr_xml_events",
    "STAGES_STAGE_MAP",
    "parse_stages_csv",
]
