"""Named preset preprocessing pipelines.

Two families of presets:

1. **Uniform presets** (single ``PreprocessingPipeline``) -- applied to every
   channel. Good for simple experiments where all modalities share the same
   treatment.

       HMCDataset(pipelines="raw")            # per-channel Resample+Bandpass
       HMCDataset(pipelines="seqsleepnet")    # same + XSleepNet spectrogram

2. **Per-modality bundle presets** (``dict`` of pipelines keyed by modality
   name) -- different treatment for EEG vs EOG vs EMG vs ECG. Bundles are
   returned as a ``dict`` that the ``BasePhysioDataset`` dispatches via its
   pipeline-resolution logic (physical-name > modality > __default__).

       HMCDataset(pipelines="time_domain")        # modality-specific filters
       HMCDataset(pipelines="time_frequency")     # time_domain + spectrogram

   The ``time_frequency`` bundle is simply ``time_domain`` with an
   ``XSleepNetSpectrogram`` step appended to each per-modality pipeline.

Users can always compose their own pipelines directly; presets are convenience
shortcuts matching the library's legacy preprocessing conventions.
"""
from typing import Dict, Union

from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps import (
    Identity,
    BandpassFilter,
    HighPassFilter,
    NotchFilter,
    Resample,
    XSleepNetSpectrogram,
)


# ---------------------------------------------------------------------------
# Uniform (single-pipeline) presets
# ---------------------------------------------------------------------------


def raw_pipeline(
    target_fs: float = 100.0,
    bp_low: float = 0.3,
    bp_high: float = 40.0,
    bp_order: int = 5,
) -> PreprocessingPipeline:
    """Minimal preprocessing: bandpass + resample (applied to every channel)."""
    return PreprocessingPipeline(
        [
            BandpassFilter(low=bp_low, high=bp_high, order=bp_order),
            Resample(target_fs=target_fs),
        ]
    )


def seqsleepnet_pipeline(
    target_fs: float = 100.0,
    bp_low: float = 0.3,
    bp_high: float = 40.0,
    bp_order: int = 5,
    nperseg: int = 200,
    noverlap: int = 100,
    nfft: int = 256,
    window: str = "hamming",
) -> PreprocessingPipeline:
    """STFT spectrogram preprocessing (mirrors legacy xsleepnet output)."""
    return PreprocessingPipeline(
        [
            BandpassFilter(low=bp_low, high=bp_high, order=bp_order),
            Resample(target_fs=target_fs),
            XSleepNetSpectrogram(
                nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window
            ),
        ]
    )


def xsleepnet_mouse_pipeline(target_fs: float = 100.0) -> PreprocessingPipeline:
    """Mouse variant: higher STFT overlap for 4-sec epochs."""
    return PreprocessingPipeline(
        [
            BandpassFilter(low=0.3, high=40.0, order=5),
            Resample(target_fs=target_fs),
            XSleepNetSpectrogram(nperseg=200, noverlap=188, nfft=256, window="hamming"),
        ]
    )


def identity_pipeline() -> PreprocessingPipeline:
    """Pure passthrough -- useful for debugging."""
    return PreprocessingPipeline([Identity()])


# ---------------------------------------------------------------------------
# Per-modality time-domain pipelines
#
# Rationale per modality (Huy Phan's code + AASM standards):
#
#   EEG -- BP 0.3-40 Hz: captures delta (0.5-4), theta (4-8), alpha (8-13),
#          sigma spindles (11-16), beta (13-30). Matches Phan exactly.
#          AASM says 0.3-35 Hz; Phan extends to 40 for DL models.
#
#   EOG -- BP 0.3-40 Hz: same as EEG. Eye movements are low-frequency.
#          Matches Phan exactly.
#
#   EMG -- HP >10 Hz: removes slow-wave crosstalk and movement artifacts
#          below 10 Hz; keeps tonic muscle activity for REM atonia detection.
#          ** Matches Phan (FIR HP 10 Hz) and AASM (10-100 Hz). **
#          At 100 Hz sampling (Nyquist = 50 Hz), effective band is 10-50 Hz.
#          Previous PhysioEx code incorrectly used BP 0.3-10 Hz (inverted!).
#
#   ECG -- BP 0.5-40 Hz: preserves QRS complex while removing baseline
#          drift. AASM recommends 0.3-70 Hz; at 100 Hz sampling the 40 Hz
#          cutoff is close to Nyquist anyway.
#
# The NotchFilter is OPTIONAL. Set notch_freq=50 or 60 to enable it.
# Most NSRR datasets are already pre-filtered at the recording amplifier;
# HMC/DCSM are raw and benefit from an explicit 50 Hz notch (EU mains).
# SeqSleepNet additionally notch-filters EMG at 50+60 Hz before the HP.
# ---------------------------------------------------------------------------


def _bandpass_steps(
    bp_low: float, bp_high: float, bp_order: int, target_fs: float, notch_freq
):
    """Build [notch?] -> bandpass -> resample for EEG/EOG/ECG."""
    steps = []
    if notch_freq is not None:
        steps.append(NotchFilter(freq=float(notch_freq), quality=30.0))
    steps.append(BandpassFilter(low=bp_low, high=bp_high, order=bp_order))
    steps.append(Resample(target_fs=target_fs))
    return steps


def _highpass_steps(cutoff: float, hp_order: int, target_fs: float, notch_freq):
    """Build [notch?] -> high-pass -> resample for EMG."""
    steps = []
    if notch_freq is not None:
        steps.append(NotchFilter(freq=float(notch_freq), quality=30.0))
    steps.append(HighPassFilter(cutoff=cutoff, order=hp_order))
    steps.append(Resample(target_fs=target_fs))
    return steps


def eeg_pipeline(
    target_fs: float = 100.0,
    notch_freq=None,
    bp_low: float = 0.3,
    bp_high: float = 40.0,
    bp_order: int = 5,
) -> PreprocessingPipeline:
    """EEG: [notch?] -> bandpass 0.3-40 Hz -> resample.  Matches Phan/AASM."""
    return PreprocessingPipeline(
        _bandpass_steps(bp_low, bp_high, bp_order, target_fs, notch_freq)
    )


def eog_pipeline(
    target_fs: float = 100.0,
    notch_freq=None,
    bp_low: float = 0.3,
    bp_high: float = 40.0,
    bp_order: int = 5,
) -> PreprocessingPipeline:
    """EOG: [notch?] -> bandpass 0.3-40 Hz -> resample.  Matches Phan/AASM."""
    return PreprocessingPipeline(
        _bandpass_steps(bp_low, bp_high, bp_order, target_fs, notch_freq)
    )


def emg_pipeline(
    target_fs: float = 100.0,
    notch_freq=None,
    hp_cutoff: float = 10.0,
    hp_order: int = 5,
) -> PreprocessingPipeline:
    """EMG (chin): [notch?] -> high-pass >10 Hz -> resample.

    Matches Huy Phan's ``fir1(Nfir, 10*2/fs, 'high')`` and the AASM
    standard (chin EMG: 10-100 Hz). At 100 Hz sampling the effective
    band is 10-50 Hz (Nyquist).
    """
    return PreprocessingPipeline(
        _highpass_steps(hp_cutoff, hp_order, target_fs, notch_freq)
    )


def ecg_pipeline(
    target_fs: float = 100.0,
    notch_freq=None,
    bp_low: float = 0.5,
    bp_high: float = 40.0,
    bp_order: int = 5,
) -> PreprocessingPipeline:
    """ECG: [notch?] -> bandpass 0.5-40 Hz -> resample."""
    return PreprocessingPipeline(
        _bandpass_steps(bp_low, bp_high, bp_order, target_fs, notch_freq)
    )


# ---------------------------------------------------------------------------
# Per-modality bundle presets (dict returned from get_preset)
# ---------------------------------------------------------------------------


def time_domain_preset(
    target_fs: float = 100.0, notch_freq=None
) -> Dict[str, PreprocessingPipeline]:
    """Per-modality time-domain preset.

    Returns a dict mapping each modality (EEG/EOG/EMG/ECG) to a modality-
    specific ``PreprocessingPipeline``. BasePhysioDataset will dispatch each
    channel to its modality's pipeline.

    ``__default__`` provides a fallback that only resamples to target_fs,
    ensuring all channels (including Resp, Temp, etc.) end up at the same rate.
    """
    return {
        "EEG": eeg_pipeline(target_fs=target_fs, notch_freq=notch_freq),
        "EOG": eog_pipeline(target_fs=target_fs, notch_freq=notch_freq),
        "EMG": emg_pipeline(target_fs=target_fs, notch_freq=notch_freq),
        "ECG": ecg_pipeline(target_fs=target_fs, notch_freq=notch_freq),
        "__default__": PreprocessingPipeline([Resample(target_fs=target_fs)]),
    }


def time_frequency_preset(
    target_fs: float = 100.0,
    notch_freq=None,
    nperseg: int = 200,
    noverlap: int = 100,
    nfft: int = 256,
    window: str = "hamming",
    clip_db: float = -25.0,
) -> Dict[str, PreprocessingPipeline]:
    """Per-modality time-frequency preset.

    Equals ``time_domain_preset`` with an ``XSleepNetSpectrogram`` step
    appended to each per-modality pipeline. The spectrogram uses power-dB
    scaling (``10 * log10(|X|^2)``) and clips at ``clip_db`` (default
    -25 dB) to remove low-power noise introduced by filter stop-band
    leakage. Output per channel is a ``(n_epochs, T, F)`` tensor suitable
    for 2D CNN / transformer models.
    """
    td = time_domain_preset(target_fs=target_fs, notch_freq=notch_freq)
    spec_step = XSleepNetSpectrogram(
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        clip_db=clip_db,
    )
    return {
        modality: PreprocessingPipeline(list(pipe.steps) + [spec_step])
        for modality, pipe in td.items()
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _foundation_pipeline(target_fs: float) -> PreprocessingPipeline:
    """Foundation model pipeline: highpass 0.5 Hz + notch 50 Hz + resample.

    Matches EEGBenchmarks' pipeline registry exactly so that cache hashes
    are compatible and preprocessed data is shared across both systems.
    """
    return PreprocessingPipeline(
        [
            HighPassFilter(cutoff=0.5, order=5),
            NotchFilter(freq=50.0, quality=30.0),
            Resample(target_fs=target_fs),
        ]
    )


def biot_pipeline() -> PreprocessingPipeline:
    """BIOT: HP 0.5 Hz + notch 50 Hz + resample to 200 Hz."""
    return _foundation_pipeline(200.0)


def bendr_pipeline() -> PreprocessingPipeline:
    """BENDR: HP 0.5 Hz + notch 50 Hz + resample to 256 Hz."""
    return _foundation_pipeline(256.0)


def cbramod_pipeline() -> PreprocessingPipeline:
    """CBraMod: HP 0.5 Hz + notch 50 Hz + resample to 200 Hz."""
    return _foundation_pipeline(200.0)


def labram_pipeline() -> PreprocessingPipeline:
    """LaBraM: HP 0.5 Hz + notch 50 Hz + resample to 200 Hz."""
    return _foundation_pipeline(200.0)


def sleepfm_pipeline() -> PreprocessingPipeline:
    """SleepFM: HP 0.5 Hz + notch 50 Hz + resample to 128 Hz."""
    return _foundation_pipeline(128.0)


def tfc_pipeline() -> PreprocessingPipeline:
    """TF-C: HP 0.5 Hz + notch 50 Hz + resample to 100 Hz."""
    return _foundation_pipeline(100.0)


def reve_pipeline() -> PreprocessingPipeline:
    """REVE: HP 0.5 Hz + notch 50 Hz + resample to 200 Hz."""
    return _foundation_pipeline(200.0)


def sjepa_pipeline() -> PreprocessingPipeline:
    """SJEPA: HP 0.5 Hz + notch 50 Hz + resample to 128 Hz."""
    return _foundation_pipeline(128.0)


def neurolm_pipeline() -> PreprocessingPipeline:
    """NeuroLM: HP 0.5 Hz + notch 50 Hz + resample to 200 Hz."""
    return _foundation_pipeline(200.0)


def coresleep_preset() -> Dict[str, PreprocessingPipeline]:
    """Per-modality spectrogram preset for CoRe-Sleep (Kontras et al. 2024).

    EEG uses the same pipeline as ``seqsleepnet`` (BP 0.3–40 Hz + STFT),
    sharing its cache.  EOG uses BP 0.3–23 Hz (paper spec) + same STFT.
    """
    return {
        "EEG": seqsleepnet_pipeline(),
        "EOG": PreprocessingPipeline(
            [
                BandpassFilter(low=0.3, high=23.0, order=5),
                Resample(target_fs=100.0),
                XSleepNetSpectrogram(
                    nperseg=200, noverlap=100, nfft=256, window="hamming"
                ),
            ]
        ),
    }


PRESETS = {
    # uniform (single-pipeline) presets
    "raw": raw_pipeline,
    "seqsleepnet": seqsleepnet_pipeline,
    "xsleepnet_mouse": xsleepnet_mouse_pipeline,
    "identity": identity_pipeline,
    # per-modality time-domain pipelines (single-pipeline form)
    "eeg": eeg_pipeline,
    "eog": eog_pipeline,
    "emg": emg_pipeline,
    "ecg": ecg_pipeline,
    # dict presets (per-modality bundles)
    "time_domain": time_domain_preset,
    "time_frequency": time_frequency_preset,
    "coresleep": coresleep_preset,
    # foundation model presets (HP 0.5Hz + Notch 50Hz + Resample, matching EEGBenchmarks)
    "biot": biot_pipeline,
    "bendr": bendr_pipeline,
    "cbramod": cbramod_pipeline,
    "labram": labram_pipeline,
    "sleepfm": sleepfm_pipeline,
    "tfc": tfc_pipeline,
    "reve": reve_pipeline,
    "sjepa": sjepa_pipeline,
    "neurolm": neurolm_pipeline,
}


def get_preset(
    name: str, **kwargs
) -> Union[PreprocessingPipeline, Dict[str, PreprocessingPipeline]]:
    """Instantiate a preset by name.

    Returns either a single ``PreprocessingPipeline`` (uniform presets) or a
    ``dict`` of modality -> pipeline (bundle presets like ``time_domain``
    and ``time_frequency``).
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name](**kwargs)


def available_presets() -> list:
    """Return sorted list of available preset names."""
    return sorted(PRESETS.keys())
