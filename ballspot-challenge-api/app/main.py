import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from app.schemas import ChallengeRequest, ChallengeResponse, FramePrediction
from app.service import (
    download_video,
    load_hot_model,
    predictions_to_frames,
    run_inference,
    video_fps,
)
from app.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_hot_model = None
_settings: Settings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor, _hot_model, _settings
    _settings = Settings()
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    logger.info("Loading checkpoint from %s …", _settings.model_checkpoint_path)
    _hot_model = load_hot_model(_settings)
    yield
    if _executor:
        _executor.shutdown(wait=False)


app = FastAPI(title="Ball spotting challenge API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _process_challenge_sync(payload: ChallengeRequest) -> ChallengeResponse:
    assert _settings is not None and _hot_model is not None
    t0 = time.perf_counter()
    url = payload.video_url
    if not url:
        raise ValueError("video_url is required (per-challenge frames mode not enabled)")

    vp = download_video(url, _settings.cache_dir, _settings.download_timeout_seconds)
    raw = run_inference(vp, _settings, _hot_model)
    fps = video_fps(vp)
    rows = predictions_to_frames(raw, fps)
    preds = [FramePrediction(frame=f, action=a, confidence=c) for f, a, c in rows]

    return ChallengeResponse(
        challenge_id=payload.challenge_id,
        predictions=preds,
        processing_time=time.perf_counter() - t0,
    )


@app.post("/challenge", response_model=ChallengeResponse)
async def challenge(req: ChallengeRequest):
    if req.frames:
        raise HTTPException(
            status_code=501,
            detail="Per-frame challenge mode is not implemented; send video_url only.",
        )
    loop = asyncio.get_running_loop()
    assert _executor is not None
    try:
        return await loop.run_in_executor(_executor, _process_challenge_sync, req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"video download failed: {exc}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=400, detail=f"video download error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("challenge failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
