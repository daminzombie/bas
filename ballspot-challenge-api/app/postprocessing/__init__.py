"""Inference → API prediction rows: compose small pure steps here as requirements grow."""

from collections.abc import Callable, Sequence
from typing import TypeAlias

from app.settings import AppConfig

from app.postprocessing.action_labels import (
    DEFAULT_ACTION_LABEL_REWRITES,
    ActionLabelRewriteStep,
)

PredictionRow: TypeAlias = tuple[int, str, float]
PostProcessFn: TypeAlias = Callable[[list[PredictionRow]], list[PredictionRow]]


def build_post_processing_pipeline(_cfg: AppConfig) -> PostProcessFn:
    """Ordered steps applied after ``predictions_to_frames`` before ``FramePrediction``."""
    steps: Sequence[PostProcessFn] = (
        ActionLabelRewriteStep(),
        # e.g. temporal smoothing, confidence floors, duplicate merging — append callables here.
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
    "PredictionRow",
    "PostProcessFn",
    "build_post_processing_pipeline",
]
