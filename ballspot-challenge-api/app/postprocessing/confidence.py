"""Confidence-based filtering for emitted raw model predictions.

This is intentionally separate from inference thresholds.  Inference thresholds decide
which model peaks become candidates; this postprocessing step removes low-confidence
rows that are likely to become unmatched scorer penalties after context cleanup.
"""

from app.postprocessing.types import PredictionRow


# Conservative per-action floors.  Frequent/low-weight actions stay relatively low so
# poor-quality videos do not become empty.  High-weight or commonly confused classes
# need stronger confidence unless another context step has already justified them.
DEFAULT_CONFIDENCE_FLOORS: dict[str, float] = {
    "pass": 0.40,
    "pass_received": 0.30,
    "free_kick": 0.30,
    "goal_kick": 0.30,
    "corner": 0.35,
    "throw_in": 0.30,
    "recovery": 0.30,
    "tackle": 0.35,
    "interception": 0.45,
    "ball_out_of_play": 0.45,
    "clearance": 0.45,
    "take_on": 0.45,
    "substitution": 0.75,
    "block": 0.50,
    "aerial_duel": 0.40,
    "shot": 0.50,
    "save": 0.60,
    "foul": 0.65,
    "goal": 0.70,
}


class PerActionConfidenceFloorStep:
    """Drop low-confidence predictions using action-specific floors."""

    __slots__ = ("_floors",)

    def __init__(self, floors: dict[str, float] | None = None) -> None:
        self._floors = dict(DEFAULT_CONFIDENCE_FLOORS)
        self._floors.update(floors or {})

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        return [
            row
            for row in rows
            if row[3] >= self._floors.get(row[1], 0.0)
        ]
