"""Physical-unit scaling steps (voltage -> mV, range -> [-1, 1])."""
import logging
import numpy as np
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class PhysicalScale(PreprocessingStep):
    """Scale a channel using its EDF header (physical dimension / range).

    Modes
    -----
    ``"mV"``
        Convert a voltage channel to millivolts from its header unit:
        ``uV``/``µV`` -> *1e-3, ``mV`` -> *1, ``V`` -> *1e3. An
        unknown or blank unit leaves the signal unchanged (with a logged
        warning), which matches the historical convention that
        unrecognised units are already in the expected unit.
    ``"range"``
        Map the header physical range to ``[-1, 1]``:
        ``y = (x - (pmax + pmin) / 2) / (|pmax - pmin| / 2)``. A zero
        range (``pmax == pmin``) leaves the signal unchanged.

    The EDF header is reached through the pipeline's optional ``header``
    argument (a dict with at least one of ``physical_dimension``,
    ``physical_min``, ``physical_max`` -- see
    :meth:`physioex.data.base.BasePhysioDataset._channel_header`). Steps
    that do not need the header ignore it. If the header is missing or
    lacks the fields the mode requires, the step logs a warning and
    passes the signal through unchanged.

    The header values are recording-specific but the step's *parameters*
    (mode) are not, so ``spec()`` (and thus ``pipeline.hash()``) covers
    the step's identity while the actual scale is fixed per cache entry
    by the source channel (cache files are keyed by physical channel).

    Parameters
    ----------
    mode : str
        ``"mV"`` or ``"range"``.
    """

    #: Keys of the channel-header dict this step consumes.
    needs_header = True

    _MV_FACTORS = {
        "uv": 1e-3,
        "µv": 1e-3,
        "mv": 1.0,
        "v": 1e3,
    }

    def __init__(self, mode: str):
        if mode not in ("mV", "range"):
            raise ValueError(f"mode must be 'mV' or 'range'; got {mode!r}")
        self.mode = mode

    def spec(self) -> str:
        return self._build_spec(mode=self.mode)

    def compile(self, fs_in: float, header: "dict | None" = None) -> CompiledStep:
        mode = self.mode
        header = header or {}

        if mode == "mV":
            unit = str(header.get("physical_dimension", "")).strip().lower()
            factor = self._MV_FACTORS.get(unit)
            if factor is None:
                logger.warning(
                    "PhysicalScale('mV'): unknown/blank physical dimension "
                    f"{unit!r}; signal passed through unchanged"
                )
                return CompiledStep(
                    apply=lambda x: np.asarray(x, dtype=np.float32),
                    fs_out=fs_in,
                )

            def apply_mV(signal: np.ndarray) -> np.ndarray:
                x = np.asarray(signal, dtype=np.float64)
                return (x * factor).astype(np.float32)

            return CompiledStep(apply=apply_mV, fs_out=fs_in)

        # mode == "range"
        pmin = header.get("physical_min")
        pmax = header.get("physical_max")
        if pmin is None or pmax is None:
            logger.warning(
                "PhysicalScale('range'): header has no physical_min/"
                "physical_max; signal passed through unchanged"
            )
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32),
                fs_out=fs_in,
            )
        pmin = float(pmin)
        pmax = float(pmax)
        half = abs(pmax - pmin) / 2.0
        center = (pmax + pmin) / 2.0
        if half == 0:
            logger.warning(
                "PhysicalScale('range'): physical range is zero "
                f"(min={pmin}, max={pmax}); signal passed through unchanged"
            )
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32),
                fs_out=fs_in,
            )

        def apply_range(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            return ((x - center) / half).astype(np.float32)

        return CompiledStep(apply=apply_range, fs_out=fs_in)
