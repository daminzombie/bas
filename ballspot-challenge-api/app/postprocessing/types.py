"""Shared postprocessing row types."""

from collections.abc import Callable
from typing import TypeAlias

PredictionRow: TypeAlias = tuple[int, str, str, float]  # (frame, action, team, confidence)
PostProcessFn: TypeAlias = Callable[[list[PredictionRow]], list[PredictionRow]]
