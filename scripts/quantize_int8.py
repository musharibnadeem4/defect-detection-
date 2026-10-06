"""OPTIONAL experiment: INT8 quantisation of artifacts/model.onnx with ONNX Runtime.

Two methods are tried: (1) dynamic quantisation (what was asked for), (2) static QDQ quantisation calibrated on
TRAIN images (fallback, because dynamic quantisation of Conv needs the ConvInteger kernel, which the CPU provider
may not implement). Each model that runs is evaluated on the VALIDATION split (never test) with the deployed
threshold kept FIXED (not re-tuned): size, logit/probability drift, decision flips, PR-AUC, recall/precision and
batch-1 CPU latency. A quantised model is only a candidate; the verdict says whether parity holds.
Usage: python scripts/quantize_int8.py   -> artifacts/model_int8_*.onnx, reports/quantization.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static  # noqa: E402
from PIL import Image  # noqa: E402

from defect_detection import preprocess as pp  # noqa: E402
from defect_detection.data import load_splits  # noqa: E402
from defect_detection.metrics import ranking_metrics, threshold_metrics  # noqa: E402
from defect_detection.utils import ROOT, load_config, load_json, positive_index, resolve, save_json  # noqa: E402


def latency(path: Path, x1: np.ndarray, name: str, threads: int, warmup: int = 20, runs: int = 200) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for i in range(warmup):
        s.run(None, {name: x1[i % len(x1)][None]})
    t = []
    for i in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: x1[i % len(x1)][None]})
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


class Reader(CalibrationDataReader):
    def __init__(self, x: np.ndarray, name: str):
        self.it = iter([{name: x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def load_x(cfg, meta, split: str, limit: int | None = None):
    sp = load_splits(cfg)
    df = sp[sp.split == split]
    if limit:
        df = df.sample(limit, random_state=cfg["project"]["seed"])
    raw = resolve(cfg, "raw_dir")
    x = pp.preprocess_batch([Image.open(raw / r.label / r.filename) for r in df.itertuples()], meta["input"]["size"],
                            meta["normalization"]["mean"], meta["normalization"]["std"])
    return x, np.array([cfg["classes"].index(l) for l in df["label"]])


def logits_of(path: Path, x: np.ndarray, name: str) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {name: x[i:i + 16]})[0] for i in range(0, len(x), 16)])


def main() -> None:
    cfg = load_config()
    meta = load_json(resolve(cfg, "model_meta"))
    fp32 = resolve(cfg, "model_onnx")
    name, thr, pos = meta["onnx"]["input_name"], meta["threshold"]["value"], positive_index(cfg)
    xv, yv = load_x(cfg, meta, "val")
    xc, _ = load_x(cfg, meta, "train", limit=100)
    base = ROOT / cfg["export"]["int8_onnx"]
    builders = {
        "dynamic": (base.with_name("model_int8_dynamic.onnx"),
                    lambda out: quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)),
        "static_qdq": (base.with_name("model_int8_static_qdq.onnx"),
                       lambda out: quantize_static(str(fp32), str(out), Reader(xc, name), quant_format=QuantFormat.QDQ,
                                                   activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)),
    }
    lg32 = logits_of(fp32, xv, name)
    p32 = pp.softmax_positive(lg32, pos)
    ref = {"pr_auc": ranking_metrics(yv, p32)["pr_auc"], "at_deployed": threshold_metrics(yv, p32, thr),
           "at_0.5": threshold_metrics(yv, p32, .5)}
    out = {"split": "validation", "n": int(len(yv)), "threshold_fixed": thr, "calibration": "100 train images (static only)",
           "fp32": {"size_mb": fp32.stat().st_size / 1e6, "pr_auc": ref["pr_auc"],
                    "recall@deployed": ref["at_deployed"]["recall"], "precision@deployed": ref["at_deployed"]["precision"],
                    "recall@0.5": ref["at_0.5"]["recall"], "precision@0.5": ref["at_0.5"]["precision"]},
           "methods": {}}
    for method, (path, build) in builders.items():
        m = {}
        try:
            build(path)
            lg = logits_of(path, xv, name)
        except Exception as e:  # noqa: BLE001
            m["status"] = f"not usable: {type(e).__name__}: {str(e)[:200]}"
            out["methods"][method] = m
            print(method, "->", m["status"])
            continue
        p = pp.softmax_positive(lg, pos)
        flips = int(((p >= thr) != (p32 >= thr)).sum())
        lat = {t: {"fp32": latency(fp32, xv, name, t), "int8": latency(path, xv, name, t)} for t in (1, 2, 4)}
        m.update(status="ok", size_mb=path.stat().st_size / 1e6, max_abs_logit_diff=float(np.abs(lg - lg32).max()),
                 mean_abs_logit_diff=float(np.abs(lg - lg32).mean()), max_abs_prob_diff=float(np.abs(p - p32).max()),
                 decision_flips_at_deployed_threshold=flips,
                 decision_flips_at_0_5=int(((p >= .5) != (p32 >= .5)).sum()),
                 pr_auc=ranking_metrics(yv, p)["pr_auc"], recall_at_deployed=threshold_metrics(yv, p, thr)["recall"],
                 precision_at_deployed=threshold_metrics(yv, p, thr)["precision"],
                 recall_at_0_5=threshold_metrics(yv, p, .5)["recall"], precision_at_0_5=threshold_metrics(yv, p, .5)["precision"],
                 latency_ms=lat, speedup_median={t: lat[t]["fp32"]["median_ms"] / lat[t]["int8"]["median_ms"] for t in lat})
        reasons = [r for r in (
            f"{flips} decision flip(s) at the deployed threshold" if flips else "",
            f"max probability drift {m['max_abs_prob_diff']:.3f} (limit 0.01)" if m["max_abs_prob_diff"] >= 0.01 else "",
            "recall at the deployed threshold drops" if m["recall_at_deployed"] < ref["at_deployed"]["recall"] else "",
            f"speed-up at most {max(m['speedup_median'].values()):.2f}x (needs > 1.2x)" if max(m["speedup_median"].values()) <= 1.2 else "") if r]
        m["adopt"] = not reasons
        m["verdict"] = "adopt" if m["adopt"] else "do NOT adopt: " + "; ".join(reasons)
        out["methods"][method] = m
    save_json(out, resolve(cfg, "reports_dir") / "quantization.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
