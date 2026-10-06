"""Service settings. Every value can be overridden with an environment variable prefixed `DEFECT_`
(e.g. DEFECT_MAX_UPLOAD_BYTES=2000000). Defaults are documented in the README."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFECT_", extra="ignore")

    artifacts_dir: Path = DEFAULT_ARTIFACTS          # must contain model.onnx + model_meta.json
    max_upload_bytes: int = 5 * 1024 * 1024          # larger uploads -> 413
    min_image_side: int = 64                         # smaller (min of width, height) -> 422
    max_image_pixels: int = 25_000_000               # decompression-bomb guard -> 422
    allowed_content_types: str = "image/jpeg,image/png,image/bmp,image/tiff"   # comma separated; others -> 415
    batch_max_images: int = 8                        # POST /predict/batch limit -> 413
    ort_intra_op_threads: int = 0                    # 0 = ONNX Runtime default
    ort_inter_op_threads: int = 0
    warmup_runs: int = 2
    log_level: str = "INFO"

    @property
    def content_types(self) -> set[str]:
        return {c.strip().lower() for c in self.allowed_content_types.split(",") if c.strip()}
