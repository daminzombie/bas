# Postprocessing Reference

This folder contains scoring-oriented cleanup that runs after raw model inference and
before the final challenge response schema is produced.

The challenge scorer penalizes unmatched predictions by action weight:

```text
score = clamp((matched_score - unmatched_penalty) / gt_total_weight, 0, 1)
```

That means a false positive for a high-weight action such as `goal`, `foul`, or
`save` can hurt much more than missing a medium-confidence event. The goal of this
postprocessing layer is therefore:

- keep valid fast soccer sequences;
- remove clear duplicate model peaks;
- remove expensive predictions that lack required soccer context;
- remove detections during dead-ball periods;
- apply final label formatting only after accuracy cleanup.

## Pipeline Order

The pipeline is assembled in `__init__.py`:

```text
SameActionTemporalDedupeStep
-> TeamConflictResolutionStep
-> GoalShotContextStep
-> SaveShotContextStep
-> FoulRestartContextStep
-> ConfusablePairResolutionStep
-> DeadBallIntervalCleanupStep
-> PerActionConfidenceFloorStep
-> ActionLabelRewriteStep
-> FinalActionTemporalDedupeStep
```

Most accuracy rules must run before final label rewriting because the rewrite maps
set pieces such as `free_kick`, `goal_kick`, `corner`, and `throw_in` to final API
`pass`. The one exception is `FinalActionTemporalDedupeStep`, which intentionally
runs after the rewrite to catch duplicates introduced by final schema formatting.

## Data Shape

Each postprocessing step receives rows shaped as:

```text
(frame, action, team, confidence)
```

Team is still available at this stage, even though the final API response drops it.
Use team where it helps avoid false merges, for example `goal` should normally be
near a same-team `shot`, while `save` should normally be near an opposite-team
`shot`.

## SameActionTemporalDedupeStep

File: `dedupe.py`

Purpose: remove repeated peaks for the same raw `(action, team)` without merging
different valid actions.

Example:

```text
frame 100: pass left 0.82
frame 102: pass left 0.76
```

These are likely duplicate peaks from one model confidence hill, so only the
highest-confidence one survives.

Counter-example:

```text
frame 100: pass left 0.82
frame 110: pass_received left 0.80
```

This is a valid fast sequence and is not deduped because the actions differ.

Tune:

- `DEFAULT_SAME_ACTION_WINDOWS`

Keep windows small for fast ball actions like `pass`, `pass_received`, and
`interception`.

## TeamConflictResolutionStep

File: `dedupe.py`

Purpose: remove same-action opposite-team duplicates before the final API drops team.

The model can predict the right action moment for both team heads. Since the final
challenge schema does not include team, keeping both usually means one can match and
the other becomes an unmatched penalty.

Example:

```text
frame 382: corner right 0.39
frame 382: corner left 0.32
```

Keep the right-team prediction and drop the lower-confidence left-team copy.

`aerial_duel` is intentionally excluded because the definition allows one event per
involved player, and close opposite-team duel predictions can be valid.

Tune:

- `DEFAULT_TEAM_CONFLICT_ACTIONS`
- `DEFAULT_SAME_ACTION_WINDOWS`

## GoalShotContextStep

File: `context.py`

Purpose: reduce high-penalty false `goal` predictions.

Domain rule: a `goal` should have a nearby same-team `shot`. Medium-confidence goals
without shot support are risky because `goal` has the highest scorer penalty.

Example dropped:

```text
frame 1000: goal left 0.62
no same-team shot nearby
```

Example kept:

```text
frame 998: shot left 0.81
frame 1000: goal left 0.66
```

High-confidence escape hatch:

```text
frame 1000: goal left 0.95
no shot nearby
```

This can still be kept because the companion `shot` may have fallen below threshold.

Tune:

- `GOAL_SHOT_WINDOW_FRAMES`
- `GOAL_KEEP_WITHOUT_SHOT_CONFIDENCE`

## SaveShotContextStep

File: `context.py`

Purpose: reduce high-penalty false `save` predictions.

Domain rule: a `save` should be near a `shot`, preferably from the opposite team.

Example dropped:

```text
frame 500: save right 0.58
no shot nearby
```

Example kept:

```text
frame 492: shot left 0.76
frame 505: save right 0.61
```

Tune:

- `SAVE_SHOT_WINDOW_FRAMES`
- `SAVE_KEEP_WITHOUT_SHOT_CONFIDENCE`

## FoulRestartContextStep

File: `context.py`

Purpose: reduce high-penalty false `foul` predictions.

Domain rule: after a foul, play usually restarts later with a dead-ball event,
especially `free_kick`. The restart may happen several seconds later, so this rule
looks forward over a broader time span instead of requiring a near-frame restart.

Example dropped:

```text
frame 300: foul 0.54
normal pass/pass_received flow continues
no future restart evidence
```

Example kept:

```text
frame 300: foul 0.64
frame 520: free_kick 0.72
```

High-confidence escape hatch:

```text
frame 300: foul 0.88
no predicted restart
```

This can still be kept because the restart may have been missed by the model.

Tune:

- `FOUL_RESTART_LOOKAHEAD_FRAMES`
- `FOUL_KEEP_WITHOUT_RESTART_CONFIDENCE`
- `FOUL_RESTART_ACTIONS`

## ConfusablePairResolutionStep

File: `context.py`

Purpose: resolve near-frame labels that usually represent one confused model
decision. This is intentionally not a generic temporal merge.

Current pairs:

- `recovery` vs `interception`
- `pass` vs `clearance`
- `block` vs `save`

### Recovery vs Interception

Definition note: `recovery` excludes active attempts to intercept the ball.

Example:

```text
frame 400: recovery right 0.62
frame 403: interception right 0.56
```

Keep `recovery` because the higher-weight `interception` is not clearly stronger.

### Pass vs Clearance

`clearance` has higher penalty and requires defensive intent. If close to a same-team
`pass`, keep `clearance` only when it is clearly stronger.

Example:

```text
frame 100: pass left 0.63
frame 102: clearance left 0.66
```

Keep `pass`, drop `clearance`.

### Pass vs Shot

A `shot` is visually a kick/contact event, so the model can also emit `pass` for the
same touch. Because `shot` has a higher scorer penalty, it must be meaningfully
stronger to beat the lower-risk `pass`.

Example:

```text
frame 508: pass right 0.46
frame 510: shot right 0.62
```

Keep `shot`, drop `pass`.

Counter-example:

```text
frame 508: pass right 0.80
frame 510: shot right 0.52
```

Keep `pass`, drop `shot`.

### Block vs Save

Both are shot-stop events. If both are near the same frame, keep the stronger one
instead of paying two high unmatched penalties.

Tune inside `_resolve_pair()` calls:

- `window_frames`
- `prefer_b_margin`
- `require_same_team`

## DeadBallIntervalCleanupStep

File: `context.py`

Purpose: remove non-game detections between a stoppage and its restart.

Supported intervals:

- `foul -> free_kick`
- `ball_out_of_play -> free_kick | throw_in | goal_kick | corner`

Events between those anchors are often players moving, setting the ball, or other
non-game activity. They can become unmatched predictions and hurt score.

Example:

```text
frame 100: foul
frame 140: pass
frame 180: pass_received
frame 260: free_kick
```

The `pass` and `pass_received` are removed because they occur during the dead-ball
interval. The `foul` and `free_kick` are preserved.

Protected actions:

- `foul`
- `ball_out_of_play`
- `free_kick`
- `throw_in`
- `goal_kick`
- `corner`
- `substitution`

`substitution` is protected but otherwise not tuned yet because it is rare and should
be handled separately once real examples are available.

Tune:

- `FOUL_DEAD_BALL_RESTART_ACTIONS`
- `BALL_OUT_RESTART_ACTIONS`
- `DEAD_BALL_PROTECTED_ACTIONS`
- `max_interval_frames`

## PerActionConfidenceFloorStep

File: `confidence.py`

Purpose: remove low-confidence emitted predictions using action-specific floors.

This is separate from inference thresholds. Inference thresholds decide which model
peaks become candidates; this postprocessing step removes rows that are likely to be
unmatched scorer penalties after context cleanup.

Example:

```text
frame 686: pass right 0.93
frame 713: pass right 0.38
```

With the default `pass` floor of `0.40`, the second low-confidence pass is removed.
This is useful when the model repeatedly calls nearby or weak touches as `pass` even
though one of them should be another class such as `interception`.

High-weight actions use higher floors because false positives are expensive:

```text
goal >= 0.70
foul >= 0.65
save >= 0.60
shot >= 0.50
```

Tune:

- `DEFAULT_CONFIDENCE_FLOORS`

Keep frequent low-weight actions relatively permissive so poor-quality videos do not
become empty, but raise floors for expensive classes that produce costly unmatched
predictions.

## ActionLabelRewriteStep

File: `action_labels.py`

Purpose: final challenge-schema formatting only.

Currently maps:

```text
free_kick -> pass
goal_kick -> pass
corner -> pass
throw_in -> pass
```

Do not move this before accuracy cleanup. Earlier steps need to know the difference
between normal passes and set-piece restarts.

## FinalActionTemporalDedupeStep

File: `dedupe.py`

Purpose: remove final-schema duplicates after label rewrite and team removal.

This catches cases where separate raw labels collapse to the same final action:

```text
frame 382: corner right -> pass
frame 382: corner left  -> pass
```

or:

```text
frame 500: throw_in -> pass
frame 502: pass     -> pass
```

It should stay after `ActionLabelRewriteStep`. Like `TeamConflictResolutionStep`, it
excludes `aerial_duel` by default.

Tune:

- `DEFAULT_FINAL_DEDUPE_ACTIONS`
- `DEFAULT_SAME_ACTION_WINDOWS`

## Future Tuning Ideas

These steps only see emitted predictions. A stronger future version could pass richer
inference metadata into postprocessing:

- dense per-class score curves;
- suppressed candidate peaks;
- second-best class margins;
- displacement-refined frame vs original peak frame;
- per-video confidence distribution.

That would allow smarter adaptive thresholds, especially for low-quality videos where
all confidences are lower than normal.
