# Build from the repository root:
#   docker build -t ballspot-challenge:latest .

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY custom-ballspotting /workspace/custom-ballspotting
RUN pip install --no-cache-dir /workspace/custom-ballspotting

COPY ballspot-challenge-api /workspace/ballspot-challenge-api
WORKDIR /workspace/ballspot-challenge-api
RUN pip install --no-cache-dir .

# Typically `*.pt` is `.dockerignored` — mount the checkpoint file at runtime to match `model_checkpoint_path` in `config/app.json`.
# Mount the checkpoint so it matches `model_checkpoint_path` in committed `config/app.json`, e.g.:
#   custom-ballspotting/checkpoints/custom_posttrain_from_custom_20260429_193215_best.pt
# Or bind-mount config/app.json. See README.md.

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
