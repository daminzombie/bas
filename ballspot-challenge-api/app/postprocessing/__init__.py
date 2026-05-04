"""Inference → API prediction rows: compose small pure steps here as requirements grow."""

from collections.abc import Sequence

from app.settings import AppConfig

from app.postprocessing.action_labels import (
    DEFAULT_ACTION_LABEL_REWRITES,
    ActionLabelRewriteStep,
)
from app.postprocessing.dedupe import (
    DEFAULT_FINAL_DEDUPE_ACTIONS,
    DEFAULT_SAME_ACTION_WINDOWS,
    DEFAULT_TEAM_CONFLICT_ACTIONS,
    FinalActionTemporalDedupeStep,
    SameActionTemporalDedupeStep,
    TeamConflictResolutionStep,
)
from app.postprocessing.context import (
    ConfusablePairResolutionStep,
    DeadBallIntervalCleanupStep,
    FoulRestartContextStep,
    GoalShotContextStep,
    SaveShotContextStep,
)
from app.postprocessing.confidence import (
    DEFAULT_CONFIDENCE_FLOORS,
    PerActionConfidenceFloorStep,
)
from app.postprocessing.types import PostProcessFn, PredictionRow


def build_post_processing_pipeline(_cfg: AppConfig) -> PostProcessFn:
    """Ordered steps applied after ``predictions_to_frames`` before ``FramePrediction``."""
    steps: Sequence[PostProcessFn] = (
        SameActionTemporalDedupeStep(),
        TeamConflictResolutionStep(),
        GoalShotContextStep(),
        SaveShotContextStep(),
        FoulRestartContextStep(),
        ConfusablePairResolutionStep(),
        DeadBallIntervalCleanupStep(),
        PerActionConfidenceFloorStep(),
        ActionLabelRewriteStep(),
        FinalActionTemporalDedupeStep(),
    )

    def run(rows: list[PredictionRow]) -> list[PredictionRow]:
        out = list(rows)
        for step in steps:
            out = step(out)
        return out

    return run


__all__ = [
    "ActionLabelRewriteStep",
    "DEFAULT_ACTION_LABEL_REWRITES",
    "DEFAULT_CONFIDENCE_FLOORS",
    "DEFAULT_FINAL_DEDUPE_ACTIONS",
    "DEFAULT_SAME_ACTION_WINDOWS",
    "DEFAULT_TEAM_CONFLICT_ACTIONS",
    "ConfusablePairResolutionStep",
    "DeadBallIntervalCleanupStep",
    "FinalActionTemporalDedupeStep",
    "FoulRestartContextStep",
    "GoalShotContextStep",
    "PerActionConfidenceFloorStep",
    "SaveShotContextStep",
    "SameActionTemporalDedupeStep",
    "TeamConflictResolutionStep",
    "PredictionRow",
    "PostProcessFn",
    "build_post_processing_pipeline",
]
