"""Fixtures for the API tests. No torch import here: these tests must run in the torch-free inference environment."""
from __future__ import annotations

import io
import json
import os
import shutil
import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from api.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "examples" / "images"

# Opt-in strict mode: with REQUIRE_ARTIFACTS=1 a missing model unit or example image FAILS the run instead of
# silently skipping every test (an all-skipped suite would look green).
STRICT = bool(os.environ.get("REQUIRE_ARTIFACTS"))
MISSING = [p.name for p in (ARTIFACTS / "model.onnx", ARTIFACTS / "model_meta.json") if not p.exists()]

need_artifacts = pytest.mark.skipif(
    bool(MISSING) and not STRICT,
    reason="artifacts/model.onnx + model_meta.json missing: run `python tasks.py export`")


@pytest.fixture(scope="session", autouse=True)
def _require_artifacts_when_strict():
    if STRICT and MISSING:
        pytest.fail(f"REQUIRE_ARTIFACTS is set but artifacts are missing: {MISSING}")


def example(prefix: str) -> Path:
    files = sorted(EXAMPLES.glob(f"{prefix}*"))
    if not files:
        msg = f"examples/images/{prefix}* missing: run scripts/predict_examples.py"
        pytest.fail(msg) if STRICT else pytest.skip(msg)
    return files[0]


def encode(img: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def png_header_claiming(width: int, height: int) -> bytes:
    """A syntactically valid PNG header that CLAIMS a huge size (a decompression-bomb stand-in with no pixel data)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")


def make_client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(**overrides)))


@pytest.fixture(scope="module")
def client():
    if not (ARTIFACTS / "model.onnx").exists():
        pytest.skip("artifacts/model.onnx missing: run `python tasks.py export`")
    with make_client() as c:
        yield c


@pytest.fixture(scope="module")
def meta():
    return json.loads((ARTIFACTS / "model_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def offline():
    """Offline reference: the same ONNX file + the torch-free preprocessing, no HTTP."""
    import onnxruntime as ort

    from defect_detection import preprocess as pp
    m = json.loads((ARTIFACTS / "model_meta.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(ARTIFACTS / m["onnx"]["file"]), providers=["CPUExecutionProvider"])
    pos = m["classes"].index(m["positive_class"])

    def run(img: Image.Image) -> float:
        x = pp.preprocess(img, m["input"]["size"], m["normalization"]["mean"], m["normalization"]["std"])
        return float(pp.softmax_positive(sess.run(None, {m["onnx"]["input_name"]: x})[0], pos)[0])
    return run


@pytest.fixture
def broken_artifacts(tmp_path):
    """Artifacts dir whose model.onnx does not match the sha256 in model_meta.json."""
    shutil.copy2(ARTIFACTS / "model_meta.json", tmp_path / "model_meta.json")
    (tmp_path / "model.onnx").write_bytes(b"not the model that was exported")
    return tmp_path
