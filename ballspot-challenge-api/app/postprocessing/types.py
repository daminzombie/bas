"""Shared postprocessing row types."""

from collections.abc import Callable
from typing import TypeAlias

# (frame, action, team, action_confidence, timestamp_ms, selected_team_confidence,
#  left_team_confidence, right_team_confidence, joint_confidence)
PredictionRow: TypeAlias = tuple[int, str, str, float, int, float, float, float, float]
PostProcessFn: TypeAlias = Callable[[list[PredictionRow]], list[PredictionRow]]
