"""Prove the torch-free preprocessing equals the training eval transform, and measure how much a "similar"
pipeline (cv2 resize, cv2 JPEG decoder) would silently change the tensors and the model output.

Compared on every test-split image (read-only): max/mean |tensor diff| vs the torchvision eval transform, and the
effect on ONNX logits / probabilities / decisions at the deployed threshold.
Usage: python scripts/check_preprocess_equivalence.py   -> reports/preprocess_equivalence.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from defect_detection import preprocess as pp  # noqa: E402
from defect_detection.data import load_splits  # noqa: E402
from defect_detection.export import _ort_session  # noqa: E402
from defect_detection.train import _datasets  # noqa: E402
from defect_detection.utils import load_config, load_json, positive_index, resolve, save_json  # noqa: E402


def main() -> None:
    cfg = load_config()
    meta = load_json(resolve(cfg, "model_meta"))
    size, thr, pos = meta["input"]["size"], meta["threshold"]["value"], positive_index(cfg)
    mean, std = np.float32(meta["normalization"]["mean"]), np.float32(meta["normalization"]["std"])
    sp = load_splits(cfg)
    df = sp[sp.split == "test"]
    raw = resolve(cfg, "raw_dir")
    paths = [raw / r.label / r.filename for r in df.itertuples()]
    _, ev = _datasets(cfg, size, load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json"), df.iloc[:1], {"x": df})
    x_tv = torch.stack([ev["x"][i][0] for i in range(len(df))]).numpy()
    sess = _ort_session(resolve(cfg, "model_onnx"))
    name = meta["onnx"]["input_name"]
    ref_logits = sess.run(None, {name: x_tv})[0]
    ref_p = pp.softmax_positive(ref_logits, pos)

    def norm_hwc(rgb_u8: np.ndarray) -> np.ndarray:  # same ToTensor/Normalize as preprocess()
        x = rgb_u8.astype(np.float32) / np.float32(255.0)
        return ((x - mean) / std).transpose(2, 0, 1)

    def pil_resize(p):  # the shipped implementation
        return pp.preprocess(Image.open(p), size, mean, std)[0]

    def cv2_resize(flag):
        def f(p):
            a = np.asarray(Image.open(p).convert("RGB"))
            return norm_hwc(cv2.resize(a, (size, size), interpolation=flag))
        return f

    def cv2_decode_pil_resize(p):  # different JPEG decoder, same (correct) resize
        a = cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rgb = Image.fromarray(a).resize((size, size), Image.BILINEAR)
        return norm_hwc(np.asarray(rgb))

    def pil_nearest_or_bicubic(resample):
        def f(p):
            return norm_hwc(np.asarray(Image.open(p).convert("RGB").resize((size, size), resample)))
        return f

    variants = {
        "shipped: PIL BILINEAR (== torchvision)": pil_resize,
        "cv2.resize INTER_LINEAR (no antialias)": cv2_resize(cv2.INTER_LINEAR),
        "cv2.resize INTER_AREA": cv2_resize(cv2.INTER_AREA),
        "cv2.resize INTER_CUBIC": cv2_resize(cv2.INTER_CUBIC),
        "PIL BICUBIC": pil_nearest_or_bicubic(Image.BICUBIC),
        "PIL NEAREST": pil_nearest_or_bicubic(Image.NEAREST),
        "cv2.imdecode + PIL BILINEAR": cv2_decode_pil_resize,
    }
    out = {"n_images": len(paths), "threshold": thr, "variants": {}}
    print(f"{len(paths)} test images, {size}px, threshold {thr:.4f}\n")
    print(f"{'variant':<42}{'max|dx|':>10}{'mean|dx|':>11}{'max|dlogit|':>13}{'max|dp|':>10}{'flips':>7}")
    for k, fn in variants.items():
        x = np.stack([fn(p) for p in paths]).astype(np.float32)
        lg = sess.run(None, {name: x})[0]
        p = pp.softmax_positive(lg, pos)
        d = np.abs(x - x_tv)
        r = {"max_abs_tensor_diff": float(d.max()), "mean_abs_tensor_diff": float(d.mean()),
             "max_abs_logit_diff": float(np.abs(lg - ref_logits).max()), "max_abs_prob_diff": float(np.abs(p - ref_p).max()),
             "decision_flips_at_threshold": int(((p >= thr) != (ref_p >= thr)).sum()),
             "decision_flips_at_0.5": int(((p >= .5) != (ref_p >= .5)).sum())}
        out["variants"][k] = r
        print(f"{k:<42}{r['max_abs_tensor_diff']:>10.4f}{r['mean_abs_tensor_diff']:>11.5f}{r['max_abs_logit_diff']:>13.4f}"
              f"{r['max_abs_prob_diff']:>10.4f}{r['decision_flips_at_threshold']:>7}")
    save_json(out, resolve(cfg, "reports_dir") / "preprocess_equivalence.json")


if __name__ == "__main__":
    main()
