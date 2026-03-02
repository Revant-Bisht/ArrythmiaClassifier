from __future__ import annotations

from pydantic import BaseModel


class FlaggedRegion(BaseModel):
    label: str
    start_s: float
    end_s: float
    peak_s: float


class Report(BaseModel):
    headline: str
    summary: str
    confidence_pct: float
    flagged_regions: list[FlaggedRegion]
    disclaimer: str


class PredictResponse(BaseModel):
    sample_id: str
    predicted_class: str
    confidence: float
    probs: dict[str, float]
    attention: list[float]
    gradcam: list[float]
    gradcam_per_class: dict[str, list[float]]
    signal_lead2: list[float]
    report: Report


class UploadRequest(BaseModel):
    signal: list[list[float]]


class UploadResponse(BaseModel):
    predicted_class: str
    confidence: float
    probs: dict[str, float]
    attention: list[float]
    signal_lead2: list[float]
    report: Report


class SampleMeta(BaseModel):
    id: str
    class_name: str
    class_full: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    cached_samples: int
