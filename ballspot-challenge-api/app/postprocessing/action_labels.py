"""Map raw model labels to challenge API taxonomy (edit mappings here only)."""


DEFAULT_ACTION_LABEL_REWRITES: dict[str, str] = {
    "free_kick": "pass",
    "goal_kick": "pass",
    "corner": "pass",
    "throw_in": "pass",
}


class ActionLabelRewriteStep:
    """Rewrite ``action`` strings using :data:`DEFAULT_ACTION_LABEL_REWRITES`."""

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table = dict(DEFAULT_ACTION_LABEL_REWRITES)

    def __call__(self, rows: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
        t = self._table
        return [(frame, t.get(action, action), conf) for frame, action, conf in rows]
