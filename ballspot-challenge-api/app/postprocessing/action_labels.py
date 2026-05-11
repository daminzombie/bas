"""Map raw model labels to challenge API taxonomy (edit mappings here only)."""

from app.postprocessing.types import PredictionRow


DEFAULT_ACTION_LABEL_REWRITES: dict[str, str] = {
    "free_kick": "pass",
    "goal_kick": "pass",
    "corner": "pass",
    "throw_in": "pass",
    "ball_out_of_play_clear": "ball_out_of_play",
    "ball_out_of_play_distant": "ball_out_of_play",
}


class ActionLabelRewriteStep:
    """Rewrite ``action`` strings using :data:`DEFAULT_ACTION_LABEL_REWRITES`."""

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table = dict(DEFAULT_ACTION_LABEL_REWRITES)

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        t = self._table
        return [(frame, t.get(action, action), conf, ts) for frame, action, conf, ts in rows]
