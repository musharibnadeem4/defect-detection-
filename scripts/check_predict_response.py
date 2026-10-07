"""Manual smoke-check script: validate a saved JSON response from POST /predict (standard library only, so it runs
without the project's dependencies).

  curl -s -X POST http://localhost:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg" -o response.json
  curl -s http://localhost:8000/model-info -o model_info.json
  python scripts/check_predict_response.py response.json [--expect defective|normal] [--model-info model_info.json]

Exits non-zero with a message on the first problem. Checks the exact key set, types, value ranges and the internal
consistency rules documented in the README (decision = p >= threshold; confidence = probability of the predicted class).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

SCHEMA = {"predicted_class": str, "confidence": float, "defect_probability": float, "threshold_used": float,
          "model_version": str, "latency_ms": float, "request_id": str, "input_quality": dict}


def fail(msg: str) -> None:
    print(f"RESPONSE CHECK FAILED: {msg}")
    sys.exit(1)


def check(body: dict, expect: str | None, info: dict | None) -> None:
    if set(body) != set(SCHEMA):
        fail(f"keys {sorted(body)} != expected {sorted(SCHEMA)}")
    for k, t in SCHEMA.items():
        v = body[k]
        if not isinstance(v, (int, float) if t is float else t) or isinstance(v, bool):
            fail(f"'{k}' has type {type(v).__name__}, expected {t.__name__}")
    p, thr, conf = body["defect_probability"], body["threshold_used"], body["confidence"]
    for name, v in (("defect_probability", p), ("threshold_used", thr), ("confidence", conf)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
            fail(f"'{name}'={v} is not a probability in [0, 1]")
    if body["latency_ms"] <= 0:
        fail("latency_ms must be positive")
    pos = "defective"
    if body["predicted_class"] not in ("normal", pos):
        fail(f"unexpected predicted_class '{body['predicted_class']}'")
    if (body["predicted_class"] == pos) != (p >= thr):
        fail(f"decision inconsistent: p={p}, threshold={thr}, class={body['predicted_class']}")
    expected_conf = p if body["predicted_class"] == pos else 1.0 - p
    if abs(conf - expected_conf) > 1e-9:
        fail(f"confidence {conf} != probability of the predicted class {expected_conf}")
    q = body["input_quality"]
    if set(q) != {"ok", "warnings", "metrics"} or not isinstance(q["ok"], bool) or not isinstance(q["warnings"], list) \
            or not all(isinstance(w, str) for w in q["warnings"]) or q["ok"] != (not q["warnings"]):
        fail(f"malformed input_quality: {q}")
    if expect and body["predicted_class"] != expect:
        fail(f"expected class '{expect}', got '{body['predicted_class']}' (p={p:.4f})")
    if info:
        if body["model_version"] != info["model_version"]:
            fail("model_version differs from /model-info")
        if abs(thr - info["threshold"]["value"]) > 1e-12:
            fail("threshold_used differs from /model-info")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("response")
    ap.add_argument("--expect", choices=["normal", "defective"])
    ap.add_argument("--model-info")
    a = ap.parse_args()
    try:
        body = json.load(open(a.response, encoding="utf-8"))
        info = json.load(open(a.model_info, encoding="utf-8")) if a.model_info else None
    except (OSError, json.JSONDecodeError) as e:
        fail(f"cannot read JSON: {e}")
    check(body, a.expect, info)
    print(f"response OK: {body['predicted_class']} (p_defect={body['defect_probability']:.4f}, "
          f"threshold={body['threshold_used']:.4f}, quality_ok={body['input_quality']['ok']}, model={body['model_version']})")


if __name__ == "__main__":
    main()
