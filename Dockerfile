# syntax=docker/dockerfile:1
# Inference image: FastAPI + ONNX Runtime, CPU only, NO torch. Build context = repo root.
#   docker build -t defect-detection-api .

# ---- builder: resolve and install the inference dependencies into a virtualenv
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install -r /tmp/requirements-inference.txt

# ---- runtime: only the venv, the API code, the torch-free preprocessing module and the model unit
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app:/app/src \
    DEFECT_ARTIFACTS_DIR=/app/artifacts
# libgomp1: OpenMP runtime used by onnxruntime on minimal Debian images (precaution)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/defect_detection/__init__.py src/defect_detection/preprocess.py /app/src/defect_detection/
COPY api /app/api
# model.onnx + model_meta.json are ONE versioned unit (the service verifies the sha256 at startup)
COPY artifacts/model.onnx artifacts/model_meta.json /app/artifacts/
USER appuser
EXPOSE 8000
# /ready = artifacts verified, session created, warm-up inference passed
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
