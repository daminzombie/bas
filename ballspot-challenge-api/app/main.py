import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from app.postprocessing import PostProcessFn, build_post_processing_pipeline
from app.postprocessing.types import PredictionRow
from app.schemas import (
    ChallengeRequest,
    ChallengeResponse,
    FramePrediction,
    RawChallengeFramePrediction,
    RawChallengeResponse,
    RawFramePrediction,
)
from app.service import (
    download_video,
    load_hot_model,
    predictions_to_frames,
    run_inference,
    video_fps,
)
from app.settings import AppConfig, load_app_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_hot_model = None
_app_config: AppConfig | None = None
_post_process: PostProcessFn | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor, _hot_model, _app_config, _post_process
    cfg, _ = load_app_config()
    _app_config = cfg
    _post_process = build_post_processing_pipeline(_app_config)
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    logger.info("Loading checkpoint from %s …", _app_config.model_checkpoint_path)
    _hot_model = load_hot_model(_app_config)
    yield
    if _executor:
        _executor.shutdown(wait=False)


app = FastAPI(title="Ball spotting challenge API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _run_challenge_pipeline(
    payload: ChallengeRequest,
) -> tuple[list[PredictionRow], list[PredictionRow], float]:
    assert _app_config is not None and _hot_model is not None and _post_process is not None
    t0 = time.perf_counter()
    url = payload.video_url
    if not url:
        raise ValueError("video_url is required (per-challenge frames mode not enabled)")

    vp = download_video(
        url, _app_config.cache_dir, _app_config.download_timeout_seconds
    )
    infer_out = run_inference(vp, _app_config, _hot_model)
    fps = video_fps(vp)
    rows_before_post = predictions_to_frames(infer_out, fps)
    rows_after_post = _post_process(list(rows_before_post))
    elapsed = time.perf_counter() - t0
    return rows_after_post, rows_before_post, elapsed


def _rows_to_challenge_predictions(rows: list[PredictionRow]) -> list[FramePrediction]:
    return [FramePrediction(frame=row[0], action=row[1], confidence=row[3]) for row in rows]


def _rows_to_raw_challenge_predictions(rows: list[PredictionRow]) -> list[RawChallengeFramePrediction]:
    return [
        RawChallengeFramePrediction(
            frame=row[0],
            action=row[1],
            confidence=row[3],
            timestamp_ms=row[4],
        )
        for row in rows
    ]


def _rows_to_raw_frame_predictions(rows: list[PredictionRow]) -> list[RawFramePrediction]:
    return [
        RawFramePrediction(
            frame=row[0],
            action=row[1],
            team=row[2],
            confidence=row[3],
            timestamp_ms=row[4],
            action_confidence=row[3],
            team_confidence=row[5],
            left_team_confidence=row[6],
            right_team_confidence=row[7],
            joint_confidence=row[8],
        )
        for row in rows
    ]


def _process_challenge_sync(payload: ChallengeRequest) -> ChallengeResponse:
    rows_after, _, elapsed = _run_challenge_pipeline(payload)
    preds = _rows_to_challenge_predictions(rows_after)
    return ChallengeResponse(
        challenge_id=payload.challenge_id,
        predictions=preds,
        processing_time=elapsed,
    )


def _process_raw_challenge_sync(payload: ChallengeRequest) -> RawChallengeResponse:
    rows_after, rows_before, elapsed = _run_challenge_pipeline(payload)
    preds = _rows_to_raw_challenge_predictions(rows_after)
    raw_preds = _rows_to_raw_frame_predictions(rows_before)
    return RawChallengeResponse(
        challenge_id=payload.challenge_id,
        predictions=preds,
        raw_predictions=raw_preds,
        processing_time=elapsed,
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


@app.post("/raw-challenge", response_model=RawChallengeResponse)
async def raw_challenge(req: ChallengeRequest):
    if req.frames:
        raise HTTPException(
            status_code=501,
            detail="Per-frame challenge mode is not implemented; send video_url only.",
        )
    loop = asyncio.get_running_loop()
    assert _executor is not None
    try:
        return await loop.run_in_executor(_executor, _process_raw_challenge_sync, req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"video download failed: {exc}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=400, detail=f"video download error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("raw-challenge failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
