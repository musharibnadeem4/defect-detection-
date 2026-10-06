"""Inference-time preprocessing with NO torch dependency (PIL / numpy / OpenCV only).

Reproduces the training eval transform (transforms.py: Resize -> ToTensor -> Normalize) exactly:

* Resize: torchvision's `Resize((s, s), BILINEAR)` on a PIL image calls `PIL.Image.resize(..., BILINEAR)`
  (PIL's bilinear filter is a convolution whose support grows when downscaling, i.e. it is
  antialiased). `cv2.resize(INTER_LINEAR)` is NOT equivalent: it samples 2x2 neighbours without
  antialiasing and gives different pixels on a 300 -> 224 downscale (see scripts/check_preprocess_equivalence.py).
  So the resize here uses PIL on purpose; do not "optimise" it to cv2.
* Channels: `convert("RGB")` exactly as in training (gray -> 3 identical channels, RGBA -> alpha dropped,
  palette -> expanded). No EXIF rotation, as in training.
* ToTensor + Normalize: float32 `x / 255`, then `(x - mean) / std` with float32 mean/std, in the same operation
  order as torchvision, so the result is bit-identical.

This module must stay importable without torch: the inference container does not ship it.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Pixel formats where `convert("RGB")` does what training did. 16-bit / float modes would be silently
# clipped by PIL, so callers reject them instead of guessing a scaling.
SUPPORTED_MODES = ("1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK")


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported pixel format '{img.mode}'")
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int, mean, std) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 3, size, size), identical to the training eval transform."""
    rgb = to_rgb(img).resize((size, size), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.uint8)                      # HWC
    x = arr.astype(np.float32) / np.float32(255.0)             # ToTensor
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)   # Normalize (float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def preprocess_batch(imgs: list[Image.Image], size: int, mean, std) -> np.ndarray:
    return np.concatenate([preprocess(i, size, mean, std) for i in imgs], axis=0)


def quality_metrics(img: Image.Image, reference_size: int | None = None) -> dict:
    """Mean brightness (0-255) and sharpness (variance of the Laplacian) of the grayscale image.

    Computed at `reference_size` (the native training resolution) so values are comparable across input
    resolutions; sharpness depends on resolution. The same function produces the training-set ranges at
    export time, so inference and training numbers are directly comparable."""
    g = to_rgb(img).convert("L")
    if reference_size and g.size != (reference_size, reference_size):
        g = g.resize((reference_size, reference_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    lap = cv2.Laplacian(a, cv2.CV_32F, ksize=1)
    return {"brightness": float(a.mean()), "sharpness": float(lap.var())}


def quality_report(metrics: dict, ranges: dict) -> dict:
    """Compare quality metrics with the training-set ranges. Informational only: it never changes a
    prediction. Wording is deliberately cautious: leaving the training range is a risk flag, not proof of
    a wrong answer."""
    warnings = []
    suffix = "input outside the training range; prediction may be less reliable"
    b, s = metrics["brightness"], metrics["sharpness"]
    bl, bh = ranges["brightness"]["low"], ranges["brightness"]["high"]
    sl, sh = ranges["sharpness"]["low"], ranges["sharpness"]["high"]
    if b < bl:
        warnings.append(f"mean brightness {b:.1f} is below the training range [{bl:.1f}, {bh:.1f}]: {suffix}")
    elif b > bh:
        warnings.append(f"mean brightness {b:.1f} is above the training range [{bl:.1f}, {bh:.1f}]: {suffix}")
    if s < sl:
        warnings.append(f"sharpness (Laplacian variance) {s:.1f} is below the training range [{sl:.1f}, {sh:.1f}] "
                        f"(image may be blurred): {suffix}")
    elif s > sh:
        warnings.append(f"sharpness (Laplacian variance) {s:.1f} is above the training range [{sl:.1f}, {sh:.1f}] "
                        f"(image may be noisy or over-sharpened): {suffix}")
    return {"ok": not warnings, "warnings": warnings,
            "metrics": {"mean_brightness": round(b, 2), "sharpness": round(s, 2)}}


def softmax_positive(logits: np.ndarray, pos: int) -> np.ndarray:
    """Probability of class `pos` from (N, C) logits (numerically stable softmax)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, pos] / e.sum(axis=1)).astype(np.float64)
