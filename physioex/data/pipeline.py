"""Preprocessing pipeline abstractions for the lazy-loading Dataset system."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class CompiledStep:
    """Result of compiling a PreprocessingStep against a specific input fs.

    Attributes:
        apply: callable(signal: np.ndarray) -> np.ndarray
        fs_out: sampling rate after this step (0 if not meaningful, e.g. spectrogram)
    """

    apply: Callable[[np.ndarray], np.ndarray]
    fs_out: float


class PreprocessingStep(ABC):
    """Base class for all preprocessing operations."""

    @abstractmethod
    def spec(self) -> str:
        """Return canonical, hashable text repr.

        Format: ClassName(key1=val1,key2=val2,...) with sorted keys, floats
        normalized via f'{float(v):.10g}'.
        """
        ...

    @abstractmethod
    def compile(
        self, fs_in: float, header: Optional[Dict[str, Any]] = None
    ) -> CompiledStep:
        """Precompute any expensive state and return a callable specialized
        for fs_in.

        ``header`` is an optional per-channel metadata dict (physical
        dimension, physical min/max, ...) forwarded to steps that declare
        they need it (``needs_header`` attribute, e.g. ``PhysicalScale``).
        Steps that ignore it simply do not read it; the signature default
        keeps every existing ``compile(fs)`` call site working.
        """
        ...

    def _format_param(self, v) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{float(v):.10g}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def _build_spec(self, **kwargs) -> str:
        name = type(self).__name__
        parts = ",".join(
            f"{k}={self._format_param(v)}" for k, v in sorted(kwargs.items())
        )
        return f"{name}({parts})"


class CompiledPipeline:
    def __init__(self, steps: List[CompiledStep], fs_out: float):
        self.steps = steps
        self.fs_out = fs_out

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for step in self.steps:
            signal = step.apply(signal)
        return signal


class PreprocessingPipeline:
    def __init__(self, steps: List[PreprocessingStep]):
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a list, got {type(steps)}")
        self.steps = list(steps)

    def spec(self) -> List[str]:
        return [s.spec() for s in self.steps]

    def hash(self) -> str:
        joined = "|".join(self.spec())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def compile(
        self, fs_in: float, header: Optional[Dict[str, Any]] = None
    ) -> CompiledPipeline:
        """Compile every step for ``fs_in``, threading the per-channel
        ``header`` (optional) through to the steps that need it."""
        compiled = []
        fs = fs_in
        for step in self.steps:
            cs = step.compile(fs, header)
            compiled.append(cs)
            if cs.fs_out > 0:  # 0 = fs-not-meaningful (e.g. spectrogram)
                fs = cs.fs_out
        return CompiledPipeline(compiled, fs_out=fs)

    def __repr__(self):
        return f"PreprocessingPipeline([{', '.join(self.spec())}])"
