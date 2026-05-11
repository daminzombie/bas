"""Shared postprocessing row types."""

from collections.abc import Callable
from typing import TypeAlias

# (frame_index, action_label, confidence, timestamp_ms)
PredictionRow: TypeAlias = tuple[int, str, float, int]
PostProcessFn: TypeAlias = Callable[[list[PredictionRow]], list[PredictionRow]]
