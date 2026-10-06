"""API tests (FastAPI TestClient, real ONNX model, no torch)."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from .conftest import (ARTIFACTS, encode, example, make_client, need_artifacts, png_header_claiming)

pytestmark = need_artifacts
SUFFIX = "input outside the training range; prediction may be less reliable"


def post_img(client, data: bytes, ctype: str = "image/png", name: str = "x.png", **kw):
    return client.post("/predict", files={"file": (name, data, ctype)}, **kw)


def post_file(client, path):
    ctype = "image/png" if path.suffix == ".png" else "image/jpeg"
    return post_img(client, path.read_bytes(), ctype, path.name)


def assert_error(r, status: int, code: str):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == r.headers["X-Request-ID"]
    assert "Traceback" not in r.text and "File \"" not in r.text           # no stack trace leaked


# ------------------------------------------------------------------ probes / info
def test_health_and_ready(client, meta):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"status": "ready", "model_version": meta["model_version"]}


def test_model_info_is_model_meta(client, meta):
    r = client.get("/model-info")
    assert r.status_code == 200 and r.json() == meta
    import hashlib
    assert hashlib.sha256((ARTIFACTS / meta["onnx"]["file"]).read_bytes()).hexdigest() == meta["onnx"]["sha256"]


# ------------------------------------------------------------------ predictions
def test_predict_defective_example(client, meta):
    r = post_file(client, example("01_defect_correct"))
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"predicted_class", "confidence", "defect_probability", "threshold_used", "model_version",
                      "latency_ms", "request_id", "input_quality"}
    assert b["predicted_class"] == "defective"
    assert b["defect_probability"] >= meta["threshold"]["value"] and b["confidence"] == b["defect_probability"]
    assert b["threshold_used"] == meta["threshold"]["value"] and b["model_version"] == meta["model_version"]
    assert isinstance(b["latency_ms"], float) and b["latency_ms"] > 0
    assert b["request_id"] == r.headers["X-Request-ID"]
    assert set(b["input_quality"]) == {"ok", "warnings", "metrics"} and b["input_quality"]["ok"] is True
    assert b["input_quality"]["warnings"] == []


def test_predict_normal_example(client):
    b = post_file(client, example("05_normal_correct")).json()
    assert b["predicted_class"] == "normal"
    assert b["defect_probability"] < b["threshold_used"]
    assert b["confidence"] == pytest.approx(1 - b["defect_probability"])


def test_decision_uses_the_threshold_not_0_5(client):
    """Anything between 0.5 and the deployed threshold must be 'normal'; confidence is then < 0.5 by definition."""
    probs = [post_file(client, p).json() for p in sorted(example("").parent.glob("*"))]
    for b in probs:
        assert (b["predicted_class"] == "defective") == (b["defect_probability"] >= b["threshold_used"])


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "1", "LA"])
def test_pixel_formats_match_offline_conversion(client, offline, mode):
    src = Image.open(example("01_defect_correct")).convert("RGB")
    img = src.convert(mode) if mode != "P" else src.convert("P", palette=Image.ADAPTIVE, colors=64)
    r = post_img(client, encode(img))
    assert r.status_code == 200, r.text
    assert r.json()["defect_probability"] == pytest.approx(offline(Image.open(io.BytesIO(encode(img)))), abs=1e-5)


def test_grayscale_equals_rgb_for_grayscale_data(client):
    """The data are grayscale stored as RGB, so an 'L' upload must give the same answer as the RGB one."""
    src = Image.open(example("05_normal_correct")).convert("RGB")
    a = post_img(client, encode(src)).json()["defect_probability"]
    b = post_img(client, encode(src.convert("L"))).json()["defect_probability"]
    assert a == pytest.approx(b, abs=1e-5)


def test_jpeg_upload(client):
    r = post_file(client, example("02_defect_correct"))
    assert r.status_code == 200 and r.json()["predicted_class"] == "defective"


# ------------------------------------------------------------------ API == offline ONNX
@pytest.mark.parametrize("prefix", ["01_defect_correct", "04_defect_correct", "05_normal_correct", "08_normal_correct",
                                    "09_oof_false_negative"])
def test_api_equals_offline_onnx(client, offline, prefix):
    path = example(prefix)
    api_p = post_file(client, path).json()["defect_probability"]
    assert api_p == pytest.approx(offline(Image.open(path)), abs=1e-5)


# ------------------------------------------------------------------ input quality (informational only)
def test_quality_warnings_for_degraded_inputs_and_prediction_unaffected(client, offline):
    blur, noise, bright = (example(f"{i}_perturbed") for i in (14, 15, 16))
    rb, rn, rr = (post_file(client, p).json() for p in (blur, noise, bright))
    for r, key in ((rb, "sharpness"), (rn, "sharpness"), (rr, "brightness")):
        assert r["input_quality"]["ok"] is False
        w = [x for x in r["input_quality"]["warnings"] if key in x]
        assert w and all(x.endswith(SUFFIX) for x in r["input_quality"]["warnings"])
    # same value the offline model gives: the quality check never changes the prediction
    for r, p in ((rb, blur), (rn, noise), (rr, bright)):
        assert r["defect_probability"] == pytest.approx(offline(Image.open(p)), abs=1e-5)


# ------------------------------------------------------------------ validation and errors
def test_corrupt_truncated_jpeg_is_400(client):
    data = example("01_defect_correct").read_bytes()
    assert_error(post_img(client, data[: len(data) // 2], "image/jpeg", "t.jpeg"), 400, "invalid_image")


def test_non_image_bytes_with_image_content_type_is_400(client):
    assert_error(post_img(client, b"this is definitely not an image", "image/jpeg"), 400, "invalid_image")


def test_non_image_file_with_text_content_type_is_415(client):
    assert_error(post_img(client, b"hello", "text/plain", "a.txt"), 415, "unsupported_media_type")


@pytest.mark.parametrize("ctype", ["application/pdf", "image/gif", "application/octet-stream"])
def test_wrong_content_type_is_415(client, ctype):
    assert_error(post_img(client, encode(Image.new("RGB", (300, 300)), "PNG"), ctype), 415, "unsupported_media_type")


def test_empty_file_is_400(client):
    assert_error(post_img(client, b"", "image/png"), 400, "empty_file")


def test_missing_file_field_is_422(client):
    assert_error(client.post("/predict"), 422, "validation_error")


def test_tiny_image_is_422():
    with make_client(min_image_side=64) as c:
        assert_error(post_img(c, encode(Image.new("RGB", (32, 300)))), 422, "image_too_small")
        assert post_img(c, encode(Image.new("RGB", (64, 64)))).status_code == 200


def test_oversize_file_is_413():
    with make_client(max_upload_bytes=2000) as c:
        noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
        data = encode(noise)
        assert len(data) > 2000
        assert_error(post_img(c, data), 413, "file_too_large")


def test_huge_declared_body_is_rejected_early_413():
    with make_client(max_upload_bytes=1000) as c:
        r = c.post("/predict", content=b"x", headers={"content-length": "5000000", "content-type": "multipart/form-data; boundary=a"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_decompression_bomb_header_is_rejected_without_decoding():
    with make_client(max_image_pixels=1_000_000) as c:
        r = post_img(c, png_header_claiming(40_000, 40_000))        # claims 1.6 gigapixels, a few dozen bytes
        assert_error(r, 422, "image_too_many_pixels")
        assert_error(post_img(c, encode(Image.new("RGB", (1200, 1200)))), 422, "image_too_many_pixels")


def test_16bit_image_is_rejected_not_silently_clipped(client):
    img = Image.fromarray(np.full((100, 100), 40000, dtype=np.uint16))
    assert img.mode in ("I;16", "I")
    assert_error(post_img(client, encode(img)), 422, "unsupported_pixel_format")


def test_unknown_route_uses_the_error_body(client):
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/predict"), 405, "method_not_allowed")


def test_unexpected_error_is_500_without_leaking_details():
    with make_client() as c:
        def boom(*a, **k):
            raise RuntimeError("secret internal detail /srv/x")
        c.app.state.service.session = type("S", (), {"run": staticmethod(boom)})()
        r = post_img(c, encode(Image.new("RGB", (300, 300), (120, 120, 120))))
        assert_error(r, 500, "internal_error")
        assert "secret" not in r.text and "RuntimeError" not in r.text


def test_request_id_is_echoed_when_safe_and_replaced_when_not(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123_X.9"})
    assert r.headers["X-Request-ID"] == "abc-123_X.9"
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["X-Request-ID"] != "bad id with spaces" and len(r.headers["X-Request-ID"]) == 32


# ------------------------------------------------------------------ readiness failure modes
def test_not_ready_when_artifacts_missing(tmp_path):
    with make_client(artifacts_dir=tmp_path) as c:
        assert c.get("/health").status_code == 200                  # liveness still ok
        assert_error(c.get("/ready"), 503, "not_ready")
        assert_error(c.get("/model-info"), 503, "not_ready")
        assert_error(post_img(c, encode(Image.new("RGB", (300, 300)))), 503, "not_ready")


def test_model_that_does_not_match_meta_is_refused(broken_artifacts):
    with make_client(artifacts_dir=broken_artifacts) as c:
        r = c.get("/ready")
        assert_error(r, 503, "not_ready")
        assert "sha256" in r.json()["error"]["message"]


# ------------------------------------------------------------------ batch
def test_batch_mixed_results(client):
    good, bad = example("01_defect_correct"), b"not an image"
    files = [("files", ("a.jpeg", good.read_bytes(), "image/jpeg")), ("files", ("b.jpeg", bad, "image/jpeg")),
             ("files", ("c.txt", b"x", "text/plain")), ("files", ("d.jpeg", example("05_normal_correct").read_bytes(), "image/jpeg"))]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["status"] for x in res] == [200, 400, 415, 200]
    assert res[0]["result"]["predicted_class"] == "defective" and res[3]["result"]["predicted_class"] == "normal"
    assert res[1]["error"]["code"] == "invalid_image" and res[2]["error"]["code"] == "unsupported_media_type"


def test_batch_limit_is_413():
    with make_client(batch_max_images=2) as c:
        data = encode(Image.new("RGB", (300, 300), (100, 100, 100)))
        files = [("files", (f"{i}.png", data, "image/png")) for i in range(3)]
        assert_error(c.post("/predict/batch", files=files), 413, "too_many_files")
