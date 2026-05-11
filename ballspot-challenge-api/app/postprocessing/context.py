"""Context-aware postprocessing for scorer-risk reduction.

These steps intentionally run on the raw model taxonomy, before the final API label
rewrite maps set pieces to ``pass``.  The scorer penalizes unmatched
predictions by action weight, so the rules below mostly remove expensive predictions
when the surrounding soccer context does not support them.
"""

from app.postprocessing.types import PredictionRow


# The API returns frame indices, not timestamps.  These defaults assume the challenge
# videos use the private 25 FPS scoring raster.  Keeping them as named constants makes
# later validation tuning straightforward.
FPS_ASSUMPTION = 25

# High-weight events should not be emitted unless the surrounding event sequence makes
# them plausible.  A very high-confidence event can still survive without support so
# low-quality videos do not lose rare true positives just because their companion event
# fell below threshold.
GOAL_SHOT_WINDOW_FRAMES = int(2.0 * FPS_ASSUMPTION)
GOAL_KEEP_WITHOUT_SHOT_CONFIDENCE = 0.92

SAVE_SHOT_WINDOW_FRAMES = int(2.0 * FPS_ASSUMPTION)
SAVE_KEEP_WITHOUT_SHOT_CONFIDENCE = 0.90

# Fouls often take several seconds to restart.  We therefore look forward for restart
# evidence instead of requiring a near-frame set piece.
FOUL_RESTART_LOOKAHEAD_FRAMES = int(15.0 * FPS_ASSUMPTION)
FOUL_KEEP_WITHOUT_RESTART_CONFIDENCE = 0.85
FOUL_RESTART_ACTIONS = frozenset({"free_kick", "ball_out_of_play"})

# Once play has stopped, model detections between the stoppage and restart are usually
# players moving/placing the ball rather than scored in-game actions.
FOUL_DEAD_BALL_RESTART_ACTIONS = frozenset({"free_kick"})
BALL_OUT_RESTART_ACTIONS = frozenset({"free_kick", "throw_in", "goal_kick", "corner"})

# Do not remove the stoppage anchor or the restart event itself.  Also leave
# substitution untouched for now per product priority; it is rare and should be tuned
# separately with real examples.
DEAD_BALL_PROTECTED_ACTIONS = frozenset(
    {
        "foul",
        "ball_out_of_play",
        "free_kick",
        "throw_in",
        "goal_kick",
        "corner",
        "substitution",
    }
)


class GoalShotContextStep:
    """Drop medium-confidence goals that do not have shot support."""

    __slots__ = ("_window", "_keep_without_shot_confidence")

    def __init__(
        self,
        window_frames: int = GOAL_SHOT_WINDOW_FRAMES,
        keep_without_shot_confidence: float = GOAL_KEEP_WITHOUT_SHOT_CONFIDENCE,
    ) -> None:
        self._window = window_frames
        self._keep_without_shot_confidence = keep_without_shot_confidence

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        shots = [row for row in rows if row[1] == "shot"]
        out: list[PredictionRow] = []
        for row in rows:
            frame, action, conf, _ts = row
            if action != "goal":
                out.append(row)
                continue

            has_nearby_shot = any(abs(shot[0] - frame) <= self._window for shot in shots)
            if has_nearby_shot or conf >= self._keep_without_shot_confidence:
                out.append(row)
        return out


class SaveShotContextStep:
    """Drop saves that are not supported by a nearby shot."""

    __slots__ = ("_window", "_keep_without_shot_confidence")

    def __init__(
        self,
        window_frames: int = SAVE_SHOT_WINDOW_FRAMES,
        keep_without_shot_confidence: float = SAVE_KEEP_WITHOUT_SHOT_CONFIDENCE,
    ) -> None:
        self._window = window_frames
        self._keep_without_shot_confidence = keep_without_shot_confidence

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        shots = [row for row in rows if row[1] == "shot"]
        out: list[PredictionRow] = []
        for row in rows:
            frame, action, conf, _ts = row
            if action != "save":
                out.append(row)
                continue

            has_any_shot = any(abs(shot[0] - frame) <= self._window for shot in shots)
            if has_any_shot or conf >= self._keep_without_shot_confidence:
                out.append(row)
        return out


class FoulRestartContextStep:
    """Keep fouls only when confidence or future restart context justifies the risk."""

    __slots__ = ("_lookahead", "_keep_without_restart_confidence", "_restart_actions")

    def __init__(
        self,
        lookahead_frames: int = FOUL_RESTART_LOOKAHEAD_FRAMES,
        keep_without_restart_confidence: float = FOUL_KEEP_WITHOUT_RESTART_CONFIDENCE,
        restart_actions: frozenset[str] = FOUL_RESTART_ACTIONS,
    ) -> None:
        self._lookahead = lookahead_frames
        self._keep_without_restart_confidence = keep_without_restart_confidence
        self._restart_actions = restart_actions

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        restarts = [row for row in rows if row[1] in self._restart_actions]
        out: list[PredictionRow] = []
        for row in rows:
            frame, action, conf, _ts = row
            if action != "foul":
                out.append(row)
                continue

            has_future_restart = any(
                frame < restart[0] <= frame + self._lookahead for restart in restarts
            )
            if has_future_restart or conf >= self._keep_without_restart_confidence:
                out.append(row)
        return out


class ConfusablePairResolutionStep:
    """Resolve near-frame labels that are usually one confused model decision."""

    __slots__ = ()

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        keep = set(range(len(rows)))
        ordered = sorted(enumerate(rows), key=lambda item: item[1][0])

        # Recovery excludes active interceptions.  For near-frame predictions,
        # keep interception only if it is clearly stronger; otherwise prefer the
        # lower-penalty recovery.
        self._resolve_pair(
            ordered,
            keep,
            action_a="recovery",
            action_b="interception",
            window_frames=8,
            prefer_b_margin=0.18,
        )

        # A received pass and recovery can be alternate explanations for the same
        # ball-control moment.  Keep whichever label is more confident.
        self._resolve_pair(
            ordered,
            keep,
            action_a="pass_received",
            action_b="recovery",
            window_frames=8,
            prefer_b_margin=0.00,
        )

        # Similarly, a pass reception and interception can be two labels for one
        # received ball.  Keep interception only when it is clearly stronger.
        self._resolve_pair(
            ordered,
            keep,
            action_a="pass_received",
            action_b="interception",
            window_frames=8,
            prefer_b_margin=0.18,
        )

        # Clearance and pass are both kicks, but clearance carries much higher
        # penalty.  Keep clearance only when it is clearly more confident.
        self._resolve_pair(
            ordered,
            keep,
            action_a="pass",
            action_b="clearance",
            window_frames=8,
            prefer_b_margin=0.20,
        )

        # A shot is visually a kick/contact event, so the model can emit a nearby
        # pass for the same touch.  Since shot is more expensive, keep it only when
        # it is meaningfully stronger; otherwise prefer the lower-risk pass.
        self._resolve_pair(
            ordered,
            keep,
            action_a="pass",
            action_b="shot",
            window_frames=6,
            prefer_b_margin=0.10,
        )

        # Block/save both require a shot.  If both are present for the same stop,
        # keep the stronger one rather than paying two high unmatched penalties.
        self._resolve_pair(
            ordered,
            keep,
            action_a="block",
            action_b="save",
            window_frames=8,
            prefer_b_margin=0.00,
        )

        return [row for idx, row in enumerate(rows) if idx in keep]

    @staticmethod
    def _resolve_pair(
        ordered: list[tuple[int, PredictionRow]],
        keep: set[int],
        *,
        action_a: str,
        action_b: str,
        window_frames: int,
        prefer_b_margin: float,
    ) -> None:
        for idx_a, row_a in ordered:
            if idx_a not in keep or row_a[1] != action_a:
                continue
            frame_a, conf_a = row_a[0], row_a[2]
            candidates_b = [
                (idx_b, row_b)
                for idx_b, row_b in ordered
                if idx_b in keep
                and row_b[1] == action_b
                and abs(row_b[0] - frame_a) <= window_frames
            ]
            if not candidates_b:
                continue
            idx_b, row_b = max(candidates_b, key=lambda item: item[1][2])
            conf_b = row_b[2]
            if conf_b >= conf_a + prefer_b_margin:
                keep.discard(idx_a)
            else:
                keep.discard(idx_b)


class DeadBallIntervalCleanupStep:
    """Remove non-game detections between a stoppage and its restart."""

    __slots__ = (
        "_foul_restart_actions",
        "_ball_out_restart_actions",
        "_protected_actions",
        "_max_interval",
    )

    def __init__(
        self,
        foul_restart_actions: frozenset[str] = FOUL_DEAD_BALL_RESTART_ACTIONS,
        ball_out_restart_actions: frozenset[str] = BALL_OUT_RESTART_ACTIONS,
        protected_actions: frozenset[str] = DEAD_BALL_PROTECTED_ACTIONS,
        max_interval_frames: int = FOUL_RESTART_LOOKAHEAD_FRAMES,
    ) -> None:
        self._foul_restart_actions = foul_restart_actions
        self._ball_out_restart_actions = ball_out_restart_actions
        self._protected_actions = protected_actions
        self._max_interval = max_interval_frames

    def __call__(self, rows: list[PredictionRow]) -> list[PredictionRow]:
        ordered = sorted(rows, key=lambda row: (row[0], -row[2]))
        intervals = self._dead_ball_intervals(ordered)
        if not intervals:
            return rows

        out: list[PredictionRow] = []
        for row in rows:
            frame, action, _conf, _ts = row
            inside_dead_ball = any(start < frame < end for start, end in intervals)
            if inside_dead_ball and action not in self._protected_actions:
                continue
            out.append(row)
        return out

    def _dead_ball_intervals(self, rows: list[PredictionRow]) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        for idx, row in enumerate(rows):
            frame, action, _conf, _ts = row
            if action == "foul":
                restart = self._first_future_restart(rows, idx, self._foul_restart_actions)
            elif action == "ball_out_of_play":
                restart = self._first_future_restart(rows, idx, self._ball_out_restart_actions)
            else:
                continue

            if restart is None:
                continue
            restart_frame = restart[0]
            if restart_frame - frame <= self._max_interval:
                intervals.append((frame, restart_frame))
        return intervals

    @staticmethod
    def _first_future_restart(
        rows: list[PredictionRow],
        start_idx: int,
        restart_actions: frozenset[str],
    ) -> PredictionRow | None:
        for row in rows[start_idx + 1 :]:
            if row[1] in restart_actions:
                return row
        return None
