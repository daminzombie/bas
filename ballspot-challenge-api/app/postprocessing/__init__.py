"""Inference → API prediction rows: compose small pure steps here as requirements grow."""

from collections.abc import Sequence

from app.settings import AppConfig

from app.postprocessing.action_labels import (
    DEFAULT_ACTION_LABEL_REWRITES,
    ActionLabelRewriteStep,
)
from app.postprocessing.dedupe import (
    DEFAULT_SAME_ACTION_WINDOWS,
    SameActionTemporalDedupeStep,
)
from app.postprocessing.context import (
    ConfusablePairResolutionStep,
    DeadBallIntervalCleanupStep,
    FoulRestartContextStep,
    GoalShotContextStep,
    SaveShotContextStep,
)
from app.postprocessing.types import PostProcessFn, PredictionRow


def build_post_processing_pipeline(_cfg: AppConfig) -> PostProcessFn:
    """Ordered steps applied after ``predictions_to_frames`` before ``FramePrediction``."""
    steps: Sequence[PostProcessFn] = (
        SameActionTemporalDedupeStep(),
        GoalShotContextStep(),
        SaveShotContextStep(),
        FoulRestartContextStep(),
        ConfusablePairResolutionStep(),
        DeadBallIntervalCleanupStep(),
        ActionLabelRewriteStep(),
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
    "DEFAULT_SAME_ACTION_WINDOWS",
    "ConfusablePairResolutionStep",
    "DeadBallIntervalCleanupStep",
    "FoulRestartContextStep",
    "GoalShotContextStep",
    "SaveShotContextStep",
    "SameActionTemporalDedupeStep",
    "PredictionRow",
    "PostProcessFn",
    "build_post_processing_pipeline",
]
