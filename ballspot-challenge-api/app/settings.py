from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    model_checkpoint_path: str

    inference_device: str | None = None
    inference_clip_frames_count: int | None = None
    inference_overlap: int | None = None
    inference_stride: int | None = None
    inference_frame_target_width: int | None = None
    inference_frame_target_height: int | None = None
    inference_val_batch_size: int | None = None
    inference_threshold: float | None = None
    inference_num_workers: int | None = None
    inference_frame_write_workers: int | None = None

    cache_dir: str = "/tmp/ballspot_challenge_cache"
    download_timeout_seconds: float = 600.0
