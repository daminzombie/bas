"""Scoring-oriented cleanup before final challenge schema formatting."""

from app.postprocessing.types import PredictionRow


DEFAULT_SAME_ACTION_WINDOWS: dict[str, int] = {
    "pass": 4,
    "pass_received": 4,
    "free_kick": 8,
    "goal_kick": 8,
    "corner": 8,
    "throw_in": 8,
    "recovery": 8,
    "tackle": 8,
    "interception": 6,
    "ball_out_of_play": 12,
    "clearance": 8,
    "take_on": 8,
    "substitution": 40,
    "block": 8,
    "aerial_duel": 6,
    "shot": 8,
    "save": 8,
    "foul": 12,
    "goal": 12,
}

# Actions that should be unique at a moment after team is ignored.  Keep
# ``aerial_duel`` out: the labeling definition allows one event per involved player,
# so opposite-team close predictions may be legitimate.
DEFAULT_TEAM_CONFLICT_ACTIONS: frozenset[str] = frozenset(
    {
        "pass",
        "pass_received",
        "free_kick",
        "goal_kick",
        "corner",
        "throw_in",
        "recovery",
        "tackle",
        "interception",
        "ball_out_of_play",
        "clearance",
        "take_on",
        "block",
        "shot",
        "save",
        "foul",
        "goal",
    }
)

DEFAULT_FINAL_DEDUPE_ACTIONS: frozenset[str] = DEFAULT_TEAM_CONFLICT_ACTIONS


class SameActionTemporalDedupeStep:
    """Remove repeated peaks for the same raw action/team without merging valid sequences."""

    __slots__ = ("_windows",)

    def __init__(self, windows: dict[str, int] | None = None) -> None:
        self._windows = dict(DEFAULT_SAME_ACTION_WINDOWS)
        self._windows.update(windows or {})

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        by_key: dict[tuple[str, str], list[PredictionRow]] = {}
        for row in rows:
            _frame, action, team, _conf = row
            by_key.setdefault((action, team), []).append(row)

        kept: list[PredictionRow] = []
        for (action, _team), action_rows in by_key.items():
            window = self._windows.get(action, 6)
            kept.extend(_nms_rows(action_rows, window))

        return sorted(kept, key=lambda row: (row[0], -row[3], row[1], row[2]))


class TeamConflictResolutionStep:
    """Resolve same-action opposite-team duplicates before team is dropped.

    The model often predicts the right action time for both team heads.  Since the
    challenge response does not carry team, keeping both creates an unmatched penalty
    for the lower-confidence copy.  This step is deliberately action-scoped and skips
    ``aerial_duel`` because multiple duel events can be valid at the same moment.
    """

    __slots__ = ("_actions", "_windows")

    def __init__(
        self,
        actions: frozenset[str] = DEFAULT_TEAM_CONFLICT_ACTIONS,
        windows: dict[str, int] | None = None,
    ) -> None:
        self._actions = actions
        self._windows = dict(DEFAULT_SAME_ACTION_WINDOWS)
        self._windows.update(windows or {})

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        kept: list[PredictionRow] = []
        by_action: dict[str, list[PredictionRow]] = {}
        for row in rows:
            _frame, action, _team, _conf = row
            if action in self._actions:
                by_action.setdefault(action, []).append(row)
            else:
                kept.append(row)

        for action, action_rows in by_action.items():
            kept.extend(_nms_rows(action_rows, self._windows.get(action, 6)))

        return sorted(kept, key=lambda row: (row[0], -row[3], row[1], row[2]))


class FinalActionTemporalDedupeStep:
    """Remove duplicates introduced by final label rewrite and team removal."""

    __slots__ = ("_actions", "_windows")

    def __init__(
        self,
        actions: frozenset[str] = DEFAULT_FINAL_DEDUPE_ACTIONS,
        windows: dict[str, int] | None = None,
    ) -> None:
        self._actions = actions
        self._windows = dict(DEFAULT_SAME_ACTION_WINDOWS)
        self._windows.update(windows or {})

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        kept: list[PredictionRow] = []
        by_action: dict[str, list[PredictionRow]] = {}
        for row in rows:
            _frame, action, _team, _conf = row
            if action in self._actions:
                by_action.setdefault(action, []).append(row)
            else:
                kept.append(row)

        for action, action_rows in by_action.items():
            kept.extend(_nms_rows(action_rows, self._windows.get(action, 6)))

        return sorted(kept, key=lambda row: (row[0], -row[3], row[1], row[2]))


def _nms_rows(rows: list[PredictionRow], window_frames: int) -> list[PredictionRow]:
    ranked = sorted(rows, key=lambda row: row[3], reverse=True)
    kept: list[PredictionRow] = []
    for row in ranked:
        frame = row[0]
        if all(abs(frame - kept_row[0]) > window_frames for kept_row in kept):
            kept.append(row)
    return kept
