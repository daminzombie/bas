# Challenge API

Fast **`POST /challenge`** service: downloads `video_url`, runs **`custom-ballspotting`**, returns per-frame-ish predictions (`frame`, `action`, `confidence`) and **`processing_time`**.

**`config/app.json`** is committed with production defaults (checkpoint path resolves **relative to the `config/` folder**, e.g. **`../../custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt`**). **`config/app.example.json`** mirrors it for documentation.

Workspace docs (Docker, checkpoints, submodule): **[`../README.md`](../README.md)**.

`frames`-based payloads validate but **`POST /challenge`** returns **501** when `frames` is set (URL-only flow for now).

The **`Dockerfile`** lives at the **repository root**; build context copies this package including **`config/app.json`** (weights are mounted at run time unless you relax `.dockerignore`).
