# Challenge API

Fast **`POST /challenge`** service: downloads `video_url`, runs **`custom-ballspotting`** (team ball action spotting), returns per-frame-ish predictions (`frame`, `action`, `confidence`) and **`processing_time`**.

**`config/app.json`** is committed with production defaults (checkpoint path resolves **relative to the `config/` folder**, e.g. **`../../custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt`**). **`config/app.example.json`** mirrors it for documentation.

Workspace docs (Docker, checkpoints, submodule): **[`../README.md`](../README.md)**.

`frames`-based payloads validate but **`POST /challenge`** returns **501** when `frames` is set (URL-only flow for now).

The **`Dockerfile`** lives at the **repository root**; build context copies this package including **`config/app.json`** (weights are mounted at run time unless you relax `.dockerignore`).

## Inference output and team information

The underlying `custom-ballspotting` model uses separate action and per-action team heads. Each spotting result carries:

- `label`
- selected `team` (`"left"` / `"right"`)
- `action_confidence`
- selected `team_confidence`
- both-side `team_confidences`
- `joint_confidence = action_confidence * team_confidence`

Internally the API pipeline propagates team and confidence details through the full postprocessing chain as:

```text
PredictionRow = (
  frame,
  action,
  team,
  action_confidence,
  timestamp_ms,
  selected_team_confidence,
  left_team_confidence,
  right_team_confidence,
  joint_confidence,
)
```

The **external `/challenge` response schema is unchanged** — `FramePrediction` still exposes only `frame`, `action`, and `confidence`, where `confidence` is the action confidence. Team is available inside the pipeline for postprocessing steps and is exposed by the raw debug response. Team conflict resolution ranks left/right duplicates with `joint_confidence` so a weak team attribution does not inflate duplicate handling.
