"""Model service: artifact loading/verification, image decoding with validation, and prediction.

Threshold, normalisation statistics, class names and quality ranges are read from model_meta.json only, so the
model, threshold and preprocessing always come from the same versioned unit (model.onnx is verified against the
sha256 recorded there).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from defect_detection import preprocess as pp

from .logging_config import LOGGER_NAME
from .settings import Settings

log = logging.getLogger(LOGGER_NAME)


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ModelService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ready = False
        self.error: str | None = None
        self.meta: dict | None = None
        self.session = None
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels  # PIL's own bomb guard (hard error at 2x)

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        """Load and verify the artifacts, create the ORT session, run warm-up inferences. Never raises: on failure
        the service stays not-ready (`error` says why) so liveness and readiness can be told apart."""
        try:
            import onnxruntime as ort
            art = self.s.artifacts_dir
            self.meta = json.loads((art / "model_meta.json").read_text(encoding="utf-8"))
            onnx_path = art / self.meta["onnx"]["file"]
            digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            if digest != self.meta["onnx"]["sha256"]:
                raise RuntimeError("model.onnx sha256 does not match model_meta.json (model/meta mismatch)")
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.s.ort_intra_op_threads
            so.inter_op_num_threads = self.s.ort_inter_op_threads
            self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            m = self.meta
            self.size = int(m["input"]["size"])
            self.classes = list(m["classes"])
            self.pos = self.classes.index(m["positive_class"])
            self.threshold = float(m["threshold"]["value"])
            self.mean, self.std = m["normalization"]["mean"], m["normalization"]["std"]
            self.ranges = m["quality_ranges"]
            self.in_name, self.out_name = m["onnx"]["input_name"], m["onnx"]["output_name"]
            self.version = m["model_version"]
            for _ in range(max(self.s.warmup_runs, 1)):   # warm-up, also validates the output
                out = self.session.run([self.out_name], {self.in_name: np.zeros((1, 3, self.size, self.size), np.float32)})[0]
            if out.shape != (1, len(self.classes)) or not np.isfinite(out).all():
                raise RuntimeError(f"warm-up inference returned an invalid output {out.shape}")
            self.ready = True
            log.info("model_loaded", extra={"fields": {
                "model_version": self.version, "sha256": digest, "threshold": self.threshold, "input_size": self.size,
                "classes": self.classes, "providers": self.session.get_providers(), "onnxruntime": ort.__version__,
                "intra_op_threads": self.s.ort_intra_op_threads, "inter_op_threads": self.s.ort_inter_op_threads}})
        except Exception as e:  # noqa: BLE001
            self.ready, self.error = False, f"{type(e).__name__}: {e}"
            log.error("model_load_failed", extra={"fields": {"error": self.error}}, exc_info=True)

    # ---------------------------------------------------------------- decoding / validation
    def decode(self, data: bytes) -> Image.Image:
        if not data:
            raise ApiError(400, "empty_file", "the uploaded file is empty")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                img = Image.open(io.BytesIO(data))               # reads the header only
                w, h = img.size
                if w * h > self.s.max_image_pixels:
                    raise ApiError(422, "image_too_many_pixels",
                                   f"image has {w * h} pixels, the limit is {self.s.max_image_pixels}")
                if min(w, h) < self.s.min_image_side:
                    raise ApiError(422, "image_too_small",
                                   f"image is {w}x{h}; the minimum side is {self.s.min_image_side} px")
                if img.mode not in pp.SUPPORTED_MODES:
                    raise ApiError(422, "unsupported_pixel_format", f"unsupported pixel format '{img.mode}'")
                img.load()                                       # full decode: catches truncated/corrupt data
        except ApiError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ApiError(422, "image_too_many_pixels", "image exceeds the maximum allowed pixel count") from None
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError):
            raise ApiError(400, "invalid_image", "the file is not a decodable image") from None
        return img

    # ---------------------------------------------------------------- prediction
    def predict(self, img: Image.Image) -> dict:
        x = pp.preprocess(img, self.size, self.mean, self.std)
        logits = self.session.run([self.out_name], {self.in_name: x})[0]
        p_def = float(pp.softmax_positive(logits, self.pos)[0])
        is_pos = p_def >= self.threshold
        neg = self.classes[1 - self.pos]
        quality = pp.quality_report(pp.quality_metrics(img, self.ranges["reference_size"]), self.ranges)
        return {"predicted_class": self.classes[self.pos] if is_pos else neg,
                "confidence": p_def if is_pos else 1.0 - p_def,
                "defect_probability": p_def, "threshold_used": self.threshold,
                "model_version": self.version, "input_quality": quality}

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict(self.decode(data))

