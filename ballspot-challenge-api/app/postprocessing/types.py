"""Shared postprocessing row types."""

from collections.abc import Callable
from typing import TypeAlias

PredictionRow: TypeAlias = tuple[int, str, str, float, int]  # (frame, action, team, confidence, timestamp_ms)
PostProcessFn: TypeAlias = Callable[[list[PredictionRow]], list[PredictionRow]]
