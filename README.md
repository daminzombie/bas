# BAS — Ball spotting workspace

Monorepo layout: **`custom-ballspotting`** is a **Git submodule** (inference library + CLI). **`ballspot-challenge-api`** is a FastAPI service that wraps inference for challenge-style HTTP payloads.

```text
bas/
├── README.md                 # this file
├── Dockerfile                # API image (CUDA PyTorch + both packages)
├── .dockerignore
├── .gitmodules
├── custom-ballspotting/      # git submodule → github.com/daminzombie/custom-ballspotting
├── ballspot-challenge-api/   # FastAPI app — config in config/app.json
```

---

## Clone

**With submodules** (recommended — you get `custom-ballspotting` checked out):

```bash
git clone --recurse-submodules <your-bas-repo-url> bas
cd bas
```

If you already cloned **without** `--recurse-submodules`:

```bash
cd bas
git submodule update --init --recursive
```

**Update the submodule** to the latest commit pinned in this repo, or enter the submodule and `git pull` as you prefer:

```bash
git submodule update --remote custom-ballspotting   # optional: track remote branch
git add custom-ballspotting
git commit -m "Bump custom-ballspotting submodule"
```

---

## Prerequisites

- **Python** ≥ 3.10
- **CUDA** + GPU PyTorch for inference (or the Docker image below)
- **`ballspot-challenge-api/config/app.json`** — committed with production-ready defaults (no checkpoint env vars)
- The **`.pt`** named in **`model_checkpoint_path`** must exist locally — typically **`custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt`** (weights are usually not committed; download before run)

---

## API configuration (`config/app.json`)

**`ballspot-challenge-api/config/app.json`** is **tracked in Git**. Paths resolve **relative to the `config/` directory** unless absolute (e.g. `../../custom-ballspotting/checkpoints/<file>.pt`).

**`config/app.example.json`** mirrors the defaults for docs or cloning as a backup template.

Startup **fails immediately** if the resolved checkpoint path is not a readable file.

**Docker:** `.dockerignore` excludes `*.pt`, so mount the checkpoint to match **`model_checkpoint_path`**:

```bash
docker run --gpus all \
  -v /path/on/host/custom_posttrain_from_custom_20260429_193215_best.pt:/workspace/custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt:ro \
  -p 8000:8000 \
  ballspot-challenge:latest
```

Optional: bind-mount **`config/app.json`** to override tuning without rebuilding.

---

## Checkpoints (`custom-ballspotting/checkpoints/`)

1. Obtain **`custom_posttrain_from_custom_20260429_193215_best.pt`** (or change **`model_checkpoint_path`** in **`config/app.json`** if you standardize another filename).
2. Place it under **`custom-ballspotting/checkpoints/`** matching the configured path before **local run** or **Docker run** (`*.pt` skipped in image build unless you relax `.dockerignore`).

---

## Run the API locally

From the **workspace root** (`bas/`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ./custom-ballspotting
pip install -e ./ballspot-challenge-api

# Place weights at custom-ballspotting/checkpoints/ per config/app.json (see Checkpoints).

cd ballspot-challenge-api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **Health:** `GET http://localhost:8000/health`
- **Challenge:** `POST http://localhost:8000/challenge` with JSON body `{"challenge_id": "…", "video_url": "https://…mp4"}`

Example:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/challenge \
  -H "Content-Type: application/json" \
  -d '{"challenge_id":"test-1","video_url":"https://example.com/clip.mp4"}' | jq .
```

Tunables (checkpoint path, batch size, frame resolution, cache dir, …) live in **`ballspot-challenge-api/config/app.json`**. **`app/settings.py`** defines parsing and defaults.

---

## Run with Docker

Build **from this repository root** (the Dockerfile lives here):

```bash
docker build -t ballspot-challenge:latest .
```

Run **with GPU**. Mount the `.pt` so it matches **`model_checkpoint_path`** in committed `config/app.json` (default basename: **`custom_posttrain_from_custom_20260429_193215_best.pt`**).

```bash
docker run --gpus all \
  -v /path/on/host/custom_posttrain_from_custom_20260429_193215_best.pt:/workspace/custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt:ro \
  -p 8000:8000 \
  ballspot-challenge:latest
```

(Optional) Bind-mount a custom `config/app.json` — see **API configuration** above.

CPU-only setups are possible but slower; use a PyTorch CPU base image if you customize the Dockerfile.

---

## Inference CLI (submodule package)

Train / infer directly from **`custom-ballspotting`**:

```bash
source .venv/bin/activate
pip install -e ./custom-ballspotting

custom-ballspotting infer-video --config custom-ballspotting/configs/inference_720p.example.json
```

See **`custom-ballspotting/README.md`** for dataset layout, training, and checkpoint metadata.

---

## Troubleshooting

| Issue | What to do |
|--------|------------|
| `custom-ballspotting` empty | Run `git submodule update --init --recursive` |
| Docker build missing submodule files | Ensure `custom-ballspotting` is checked out **before** `docker build` (clone with `--recurse-submodules`) |
| Checkpoint not found | Fix `model_checkpoint_path` in `config/app.json` and ensure that file exists |
| CUDA OOM | Lower `inference_val_batch_size` in `app.json` |
