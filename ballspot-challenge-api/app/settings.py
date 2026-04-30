"""Load API settings from JSON (no env vars required for checkpoints)."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ballspot-challenge-api/ (must contain config/app.json alongside this package tree)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REL_CONFIG = Path("config") / "app.json"


class AppConfig(BaseModel):
    """API config loaded from ``config/app.json``.

    Omit optional inference keys → checkpoint sibling ``*.metadata.json`` then
    ``custom-ballspotting`` defaults. For API deploys aligned with ``configs/inference_720p.example.json``,
    commit explicit values under ``ballspot-challenge-api/config/app.json``.
    """

    model_checkpoint_path: str = Field(..., min_length=1)
    inference_device: str | None = None
    inference_clip_frames_count: int | None = None
    inference_overlap: int | None = None
    inference_stride: int | None = None
    inference_frame_target_width: int | None = None
    inference_frame_target_height: int | None = None
    inference_features_model_name: str | None = None
    inference_temporal_shift_mode: str | None = None
    inference_n_layers: int | None = None
    inference_sgp_ks: int | None = None
    inference_sgp_k: int | None = None
    inference_gaussian_blur_kernel_size: int | None = None
    inference_val_batch_size: int | None = None
    inference_threshold: float | None = None
    inference_num_workers: int | None = None
    inference_frame_write_workers: int | None = None

    cache_dir: str = "/tmp/ballspot_challenge_cache"
    download_timeout_seconds: float = 600.0


def resolve_config_relative(path_str: str, *, config_dir: Path) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((config_dir / p).resolve())


def load_app_config(config_file: Path | None = None) -> tuple[AppConfig, Path]:
    """
    Read ``ballspot-challenge-api/config/app.json`` (or an explicit path).
    Paths in JSON are resolved **relative to the directory that contains ``app.json``**
    (i.e. the ``config/`` folder), unless absolute.
    """
    path = (config_file or (PACKAGE_ROOT / DEFAULT_REL_CONFIG)).resolve()
    if not path.is_file():
        example = path.parent / "app.example.json"
        hint = f" Copy {example.name} to {path.name} under {path.parent}." if example.is_file() else ""
        raise FileNotFoundError(f"Missing API config: {path}.{hint}")

    logger.info("Loading API config from %s", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = AppConfig.model_validate(data)
    cfg_dir = path.parent.resolve()

    resolved = cfg.model_copy(
        update={
            "model_checkpoint_path": resolve_config_relative(
                cfg.model_checkpoint_path, config_dir=cfg_dir
            ),
            "cache_dir": resolve_config_relative(cfg.cache_dir, config_dir=cfg_dir),
        }
    )

    ckpt_path = resolved.model_checkpoint_path
    if not Path(ckpt_path).is_file():
        raise RuntimeError(
            f"Configured model_checkpoint_path is not an existing file: {ckpt_path!r} "
            f"(from JSON {path}); place the .pt on disk (e.g. under custom-ballspotting/checkpoints/ "
            "per config/app.json) or update the path."
        )

    cache_path = resolved.cache_dir
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    return resolved, path
