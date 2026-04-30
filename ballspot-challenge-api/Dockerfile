# Build from parent directory containing both repos, e.g.:
#   cd ..   # cd to folder that has custom-ballspotting/ and ballspot-challenge-api/
#   docker build -f ballspot-challenge-api/Dockerfile -t ballspot-challenge:latest .

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

EXPOSE 8000

# Set at runtime (or bake a COPY checkpoint into the image).
ENV MODEL_CHECKPOINT_PATH=""
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
