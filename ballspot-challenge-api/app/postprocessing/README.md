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
-> PerActionConfidenceFloorStep
-> ConfusablePairResolutionStep
-> GoalShotContextStep
-> SaveShotContextStep
-> FoulRestartContextStep
-> DeadBallIntervalCleanupStep
-> ActionLabelRewriteStep
-> FinalActionTemporalDedupeStep
```

The confidence floor and confusable-pair pass run before context rules so goal/save/foul
support is based on predictions that are still plausible output candidates. Most
accuracy rules still run before final label rewriting because the rewrite maps set
pieces such as `free_kick`, `goal_kick`, `corner`, and `throw_in` to final API `pass`.
The one exception is `FinalActionTemporalDedupeStep`, which intentionally runs after
the rewrite to catch duplicates introduced by final schema formatting.

## Data Shape

Each postprocessing step receives rows shaped as:

```text
(frame, action, confidence, timestamp_ms)
```

Context and confidence steps use **`confidence`** (action softmax at the decoded peak).
Temporal dedupe ranks candidates by **`confidence`** within per-action windows.

## SameActionTemporalDedupeStep

File: `dedupe.py`

Purpose: remove repeated peaks for the same raw `action` without merging different
valid actions.

Example:

```text
frame 100: pass 0.82
frame 102: pass 0.76
```

These are likely duplicate peaks from one model confidence hill, so only the
highest-confidence one survives.

Counter-example:

```text
frame 100: pass 0.82
frame 110: pass_received 0.80
```

This is a valid fast sequence and is not deduped because the actions differ.

Tune:

- `DEFAULT_SAME_ACTION_WINDOWS`

Keep windows small for fast ball actions like `pass`, `pass_received`, and
`interception`.

## GoalShotContextStep

File: `context.py`

Purpose: reduce high-penalty false `goal` predictions.

Domain rule: a `goal` should have a nearby `shot`. Medium-confidence goals without
shot support are risky because `goal` has the highest scorer penalty.

Example dropped:

```text
frame 1000: goal 0.62
no shot nearby
```

Example kept:

```text
frame 998: shot 0.81
frame 1000: goal 0.66
```

High-confidence escape hatch:

```text
frame 1000: goal 0.95
no shot nearby
```

This can still be kept because the companion `shot` may have fallen below threshold.

Tune:

- `GOAL_SHOT_WINDOW_FRAMES`
- `GOAL_KEEP_WITHOUT_SHOT_CONFIDENCE`

## SaveShotContextStep

File: `context.py`

Purpose: reduce high-penalty false `save` predictions.

Domain rule: a `save` should be near a `shot`.

Example dropped:

```text
frame 500: save 0.58
no shot nearby
```

Example kept:

```text
frame 492: shot 0.76
frame 505: save 0.61
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
- `pass_received` vs `recovery`
- `pass_received` vs `interception`
- `pass` vs `clearance`
- `pass` vs `shot`
- `block` vs `save`

### Recovery vs Interception

Definition note: `recovery` excludes active attempts to intercept the ball.

Example:

```text
frame 400: recovery right 0.62
frame 403: interception right 0.56
```

Keep `recovery` because the higher-weight `interception` is not clearly stronger.

### Pass Received vs Recovery

Both labels can describe a near-frame ball-control moment. If a `pass_received` and
`recovery` occur within 8 frames, keep the higher-confidence action.

Example:

```text
frame 220: pass_received 0.57
frame 226: recovery 0.64
```

Keep `recovery`, drop `pass_received`.

### Pass Received vs Interception

Like `recovery`, `pass_received` can conflict with `interception` around the same
received-ball moment. Since `interception` carries higher scorer risk, keep it only
when it is clearly stronger.

Example:

```text
frame 220: pass_received 0.58
frame 226: interception 0.66
```

Keep `pass_received`, drop `interception`.

### Pass vs Clearance

`clearance` has higher penalty and requires defensive intent. If it is close to a
`pass`, keep `clearance` only when it is clearly stronger.

Example:

```text
frame 100: pass 0.63
frame 102: clearance 0.66
```

Keep `pass`, drop `clearance`.

### Pass vs Shot

A `shot` is visually a kick/contact event, so the model can also emit `pass` for the
same touch. Because `shot` has a higher scorer penalty, it must be meaningfully
stronger to beat the lower-risk `pass`.

Example:

```text
frame 508: pass 0.46
frame 510: shot 0.62
```

Keep `shot`, drop `pass`.

Counter-example:

```text
frame 508: pass 0.80
frame 510: shot 0.52
```

Keep `pass`, drop `shot`.

### Block vs Save

Both are shot-stop events. If both are near the same frame, keep the stronger one
instead of paying two high unmatched penalties.

Tune inside `_resolve_pair()` calls:

- `window_frames`
- `prefer_b_margin`

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

Purpose: remove final-schema duplicates after label rewrite.

This catches cases where separate raw labels collapse to the same final action:

```text
frame 382: corner -> pass
frame 383: corner -> pass
```

or:

```text
frame 500: throw_in -> pass
frame 502: pass     -> pass
```

It should stay after `ActionLabelRewriteStep`. It excludes `aerial_duel` by default.

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
