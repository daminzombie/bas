"""Download video URLs and wire custom-ballspotting inference."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import cv2
import httpx

from custom_ballspotting.actions import NUM_ACTION_CLASSES
from custom_ballspotting.inference import infer_video, resolve_infer_video_params
from custom_ballspotting.model.tdeed import CustomTDeedModule

from app.settings import AppConfig

logger = logging.getLogger(__name__)


def _url_digest(url: str) -> str:
    return hashlib.sha256(url.strip().encode()).hexdigest()[:32]


def download_video(url: str, cache_dir: str, timeout: float) -> str:
    key = _url_digest(url)
    dst_dir = Path(cache_dir) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    video_path = dst_dir / "video.mp4"
    if video_path.exists() and video_path.stat().st_size > 0:
        return str(video_path.resolve())

    tmp = tempfile.NamedTemporaryFile(
        prefix="dl_", suffix=".mp4.part", dir=dst_dir, delete=False
    )
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        logger.info("Downloading video …")
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with tmp_path.open("wb") as out:
                for chunk in r.iter_bytes(1024 * 1024):
                    out.write(chunk)
        os.replace(tmp_path, video_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return str(video_path.resolve())


def video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 25.0
    finally:
        cap.release()


def predictions_to_frames(raw: dict, fps: float) -> list[tuple[int, str, float]]:
    out: list[tuple[int, str, float]] = []
    for p in raw.get("predictions", []):
        pos_ms = int(p["position"])
        frame = max(0, int(round(pos_ms / 1000.0 * fps)))
        confidence = float(p["confidence"])
        confidence = max(0.0, min(1.0, confidence))
        out.append((frame, str(p["label"]), confidence))
    out.sort(key=lambda t: (t[0], -t[2]))
    return out


def build_infer_kw(cfg: AppConfig) -> dict[str, object]:
    return {
        "clip_frames_count": cfg.inference_clip_frames_count,
        "overlap": cfg.inference_overlap,
        "stride": cfg.inference_stride,
        "frame_target_width": cfg.inference_frame_target_width,
        "frame_target_height": cfg.inference_frame_target_height,
        "features_model_name": cfg.inference_features_model_name,
        "temporal_shift_mode": cfg.inference_temporal_shift_mode,
        "n_layers": cfg.inference_n_layers,
        "sgp_ks": cfg.inference_sgp_ks,
        "sgp_k": cfg.inference_sgp_k,
        "gaussian_blur_kernel_size": cfg.inference_gaussian_blur_kernel_size,
        "val_batch_size": cfg.inference_val_batch_size,
        "inference_threshold": cfg.inference_threshold,
        "device": cfg.inference_device,
        "num_workers": cfg.inference_num_workers if cfg.inference_num_workers is not None else 0,
        "frame_write_workers": cfg.inference_frame_write_workers
        if cfg.inference_frame_write_workers is not None
        else 8,
    }


def resolve_merged_params(cfg: AppConfig) -> dict:
    kw = build_infer_kw(cfg)
    return resolve_infer_video_params(
        cfg.model_checkpoint_path,
        clip_frames_count=kw["clip_frames_count"],
        overlap=kw["overlap"],
        stride=kw["stride"],
        frame_target_width=kw["frame_target_width"],
        frame_target_height=kw["frame_target_height"],
        features_model_name=kw["features_model_name"],
        temporal_shift_mode=kw["temporal_shift_mode"],
        n_layers=kw["n_layers"],
        sgp_ks=kw["sgp_ks"],
        sgp_k=kw["sgp_k"],
        gaussian_blur_kernel_size=kw["gaussian_blur_kernel_size"],
        val_batch_size=kw["val_batch_size"],
        inference_threshold=kw["inference_threshold"],
        device=kw["device"],
        extract_frames=False,
    )


def load_hot_model(cfg: AppConfig):
    """Load weights once + one warm-up forward (matches ``infer_video`` geometry)."""
    import torch

    merged = resolve_merged_params(cfg)

    device = merged["device"]
    model = CustomTDeedModule(
        clip_len=merged["clip_frames_count"],
        num_actions=NUM_ACTION_CLASSES,
        n_layers=merged["n_layers"],
        sgp_ks=merged["sgp_ks"],
        sgp_k=merged["sgp_k"],
        features_model_name=merged["features_model_name"],
        temporal_shift_mode=merged["temporal_shift_mode"],
        gaussian_blur_ks=merged["gaussian_blur_kernel_size"],
    )
    model.load_all(cfg.model_checkpoint_path)
    model.to(device)
    model.eval()

    dummy = torch.zeros(
        1,
        merged["clip_frames_count"],
        3,
        merged["frame_target_height"],
        merged["frame_target_width"],
        device=device,
    )
    with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=device == "cuda"):
        model(dummy, inference=True)
    if device == "cuda":
        torch.cuda.synchronize()
    logger.info("Hot model ready on %s.", device)
    return model


def run_inference(video_path: str, cfg: AppConfig, hot_model) -> dict:
    kw = build_infer_kw(cfg)
    return infer_video(
        video_path=video_path,
        model_checkpoint_path=cfg.model_checkpoint_path,
        output_path=None,
        model=hot_model,
        extract_frames=True,
        **kw,
    )
