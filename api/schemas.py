"""Response models (they also drive the OpenAPI docs)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class InputQuality(BaseModel):
    ok: bool
    warnings: list[str]
    metrics: dict[str, float] = Field(description="measured mean brightness and sharpness (Laplacian variance) "
                                                  "at the training reference resolution")


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(description="softmax probability of the predicted class. Because the decision "
                                          "threshold is not 0.5, a 'normal' prediction can have confidence < 0.5")
    defect_probability: float
    threshold_used: float
    model_version: str
    latency_ms: float = Field(description="server-side time for decode + preprocess + inference")
    request_id: str
    input_quality: InputQuality


class BatchItem(BaseModel):
    filename: str | None
    status: int
    result: dict | None = None
    error: dict | None = None


class BatchResponse(BaseModel):
    model_version: str
    latency_ms: float
    request_id: str
    results: list[BatchItem]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
