# Challenge API

Fast **`POST /challenge`** service: downloads `video_url`, runs **`custom-ballspotting`** (team ball action spotting), returns per-frame-ish predictions (`frame`, `action`, `confidence`) and **`processing_time`**.

**`config/app.json`** is committed with production defaults (checkpoint path resolves **relative to the `config/` folder**, e.g. **`../../custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt`**). **`config/app.example.json`** mirrors it for documentation.

Workspace docs (Docker, checkpoints, submodule): **[`../README.md`](../README.md)**.

`frames`-based payloads validate but **`POST /challenge`** returns **501** when `frames` is set (URL-only flow for now).

The **`Dockerfile`** lives at the **repository root**; build context copies this package including **`config/app.json`** (weights are mounted at run time unless you relax `.dockerignore`).

## Inference output and team information

The underlying `custom-ballspotting` model now produces **team-aware** dudek/SoccerNet BAS predictions: each spotting result carries both an `action` label such as `PASS` / `SHOT` / `FREE KICK` and a `team` (`"left"` / `"right"`). Internally the API pipeline propagates team as a `PredictionRow = (frame, action, team, confidence)` 4-tuple.

The **external API response schema is unchanged** — `FramePrediction` still exposes only `frame`, `action`, and `confidence`. Team is available in `/raw-challenge`. By default `enable_postprocessing` is `false`, so `/challenge` returns the decoded model labels without the older custom action rewrites/cleanup pipeline. Set it to `true` only if you intentionally want those extra API-side steps.
