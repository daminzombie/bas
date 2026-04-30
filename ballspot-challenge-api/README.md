# Challenge API

FastAPI `/challenge` endpoint that downloads a soccer clip URL and runs `custom-ballspotting` inference.

## Layout

Place this repo **beside** `custom-ballspotting` (sibling folders). The Docker image copies both directories in one build.

```text
parent/
├── custom-ballspotting/     # inference package (not on PyPI)
└── ballspot-challenge-api/  # this service
```

## Local dev

```bash
cd custom-ballspotting && pip install -e .
cd ../ballspot-challenge-api && pip install -e .

export MODEL_CHECKPOINT_PATH=/path/to/best.pt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Optional env vars: `INFERENCE_FRAME_TARGET_HEIGHT`, `INFERENCE_VAL_BATCH_SIZE`, `CACHE_DIR`, etc. (see `app/settings.py`).

## Docker

From the parent directory that contains **both** folders:

```bash
docker build -f ballspot-challenge-api/Dockerfile -t ballspot-challenge:latest .
docker run --gpus all -e MODEL_CHECKPOINT_PATH=/weights/best.pt -v /your/weights:/weights -p 8000:8000 ballspot-challenge:latest
```

## Request

POST `/challenge`

```json
{
  "challenge_id": "abc",
  "video_url": "https://example.com/clip.mp4"
}
```
