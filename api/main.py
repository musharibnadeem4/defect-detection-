"""FastAPI inference service (ONNX Runtime, CPU).

  POST /predict        multipart image upload (field `file`) -> prediction JSON
  POST /predict/batch  up to DEFECT_BATCH_MAX_IMAGES images (field `files`), per-image result or error
  GET  /health         liveness (process is up)
  GET  /ready          readiness (artifacts verified, ONNX session created, warm-up inference passed)
  GET  /model-info     contents of model_meta.json

All errors share one body: {"error": {"code", "message", "request_id"}}. Settings: api/settings.py (env DEFECT_*).
"""
from __future__ import annotations

import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_config import setup_logging
from .schemas import BatchResponse, ErrorResponse, PredictResponse
from .service import ApiError, ModelService
from .settings import Settings

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
QUIET_PATHS = {"/health", "/ready"}          # probes: logged at DEBUG to keep the log readable
ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logger = setup_logging(settings.log_level)
    service = ModelService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("startup", extra={"fields": {"config": settings.model_dump(mode="json")}})
        await run_in_threadpool(service.load)
        logger.info("ready" if service.ready else "not_ready",
                    extra={"fields": {"model_version": getattr(service, "version", None), "error": service.error}})
        yield
        logger.info("shutdown")

    app = FastAPI(title="Defect detection API", version="1.0", lifespan=lifespan)
    app.state.service, app.state.settings = service, settings

    def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        request.state.error_code = code
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": rid}})

    # ------------------------------------------------------------ middleware: request id, size guard, access log
    @app.middleware("http")
    async def context(request: Request, call_next):
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id", "")
        request.state.request_id = rid if REQUEST_ID_RE.match(rid) else uuid.uuid4().hex   # no log injection
        limit = settings.max_upload_bytes * (settings.batch_max_images if request.url.path.endswith("/batch") else 1)
        cl = request.headers.get("content-length", "")
        try:
            if cl.isdigit() and int(cl) > limit + 64 * 1024:     # cheap early reject (multipart overhead allowed)
                response = error_response(request, 413, "file_too_large",
                                          f"request body exceeds the limit of {settings.max_upload_bytes} bytes per file")
            else:
                response = await call_next(request)
        except Exception:  # noqa: BLE001 - last resort: never leak a stack trace to the client
            logger.exception("unhandled_error", extra={"fields": {"request_id": request.state.request_id,
                                                                  "path": request.url.path}})
            response = error_response(request, 500, "internal_error", "unexpected server error")
        response.headers["X-Request-ID"] = request.state.request_id
        st = request.state
        fields = {"request_id": st.request_id, "method": request.method, "path": request.url.path,
                  "status": response.status_code, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                  "predicted_class": getattr(st, "predicted_class", None),
                  "defect_probability": getattr(st, "defect_probability", None),
                  "quality_warnings": getattr(st, "quality_warnings", None),
                  "error_code": getattr(st, "error_code", None)}
        if hasattr(st, "n_images"):
            fields["n_images"] = st.n_images
        logger.log(10 if request.url.path in QUIET_PATHS and response.status_code < 400 else 20, "request",
                   extra={"fields": {k: v for k, v in fields.items() if v is not None}})
        return response

    # ------------------------------------------------------------ error handlers (one body shape)
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return error_response(request, exc.status, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed"}
        return error_response(request, exc.status_code, codes.get(exc.status_code, f"http_{exc.status_code}"),
                              str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        return error_response(request, 422, "validation_error",
                              f"missing or invalid request field(s): {', '.join(fields) or 'body'}")

    # ------------------------------------------------------------ helpers
    def require_ready() -> None:
        if not service.ready:
            raise ApiError(503, "not_ready", service.error or "model is not loaded")

    async def read_upload(file: UploadFile) -> bytes:
        ct = (file.content_type or "").split(";")[0].strip().lower()
        if ct not in settings.content_types:
            raise ApiError(415, "unsupported_media_type",
                           f"content type '{ct or 'missing'}' is not supported; allowed: "
                           f"{', '.join(sorted(settings.content_types))}")
        data = await file.read(settings.max_upload_bytes + 1)      # never buffers more than limit + 1 byte
        if len(data) > settings.max_upload_bytes:
            raise ApiError(413, "file_too_large", f"file exceeds the limit of {settings.max_upload_bytes} bytes")
        return data

    async def run_prediction(data: bytes, request: Request) -> dict:
        t0 = time.perf_counter()
        try:
            res = await run_in_threadpool(service.predict_bytes, data)
        except ApiError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("predict_failed", extra={"fields": {"request_id": request.state.request_id}})
            raise ApiError(500, "internal_error", "unexpected server error") from None
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return res

    # ------------------------------------------------------------ routes
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        require_ready()
        return {"status": "ready", "model_version": service.version}

    @app.get("/model-info")
    async def model_info():
        require_ready()
        return service.meta

    @app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
    async def predict(request: Request, file: UploadFile = File(...)):
        require_ready()
        res = await run_prediction(await read_upload(file), request)
        res["request_id"] = request.state.request_id
        request.state.predicted_class = res["predicted_class"]
        request.state.defect_probability = round(res["defect_probability"], 6)
        request.state.quality_warnings = res["input_quality"]["warnings"] or None
        return res

    @app.post("/predict/batch", response_model=BatchResponse, responses=ERROR_RESPONSES)
    async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
        require_ready()
        if len(files) > settings.batch_max_images:
            raise ApiError(413, "too_many_files", f"at most {settings.batch_max_images} images per batch")
        t0, results = time.perf_counter(), []
        for f in files:
            try:
                res = await run_prediction(await read_upload(f), request)
                res.pop("model_version")
                results.append({"filename": f.filename, "status": 200, "result": res})
            except ApiError as e:
                results.append({"filename": f.filename, "status": e.status, "error": {"code": e.code, "message": e.message}})
        request.state.n_images = len(files)
        return {"model_version": service.version, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "request_id": request.state.request_id, "results": results}

    return app


app = create_app()
