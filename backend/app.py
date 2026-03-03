from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import ONNXSession
from backend.models import (
    HealthResponse,
    PredictResponse,
    SampleMeta,
    UploadRequest,
    UploadResponse,
)
from backend.reports import generate_report

CACHE_DIR = Path(__file__).parent / "cache"
ONNX_PATH = Path(__file__).parent.parent / "checkpoints" / "model.onnx"

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_FULL = {
    "NORM": "Normal",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T-wave Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

_cache: dict[str, dict] = {}
_session: ONNXSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session
    _session = ONNXSession(str(ONNX_PATH))
    for path in sorted(CACHE_DIR.glob("sample_*.json")):
        class_name = path.stem.removeprefix("sample_")
        with open(path) as f:
            _cache[class_name] = json.load(f)
    yield


app = FastAPI(
    title="Arrhythmia Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", cached_samples=len(_cache))


@app.get("/samples", response_model=list[SampleMeta])
async def list_samples() -> list[SampleMeta]:
    return [
        SampleMeta(
            id=name,
            class_name=name,
            class_full=CLASS_FULL[name],
            confidence=_cache[name]["confidence"],
        )
        for name in CLASS_NAMES
        if name in _cache
    ]


@app.get("/predict/preloaded/{class_name}", response_model=PredictResponse)
async def predict_preloaded(class_name: str) -> PredictResponse:
    key = class_name.upper()
    if key not in _cache:
        raise HTTPException(status_code=404, detail=f"No cached sample for {key}")
    return PredictResponse(**_cache[key])


@app.post("/predict/upload", response_model=UploadResponse)
async def predict_upload(body: UploadRequest) -> UploadResponse:
    arr = np.array(body.signal, dtype=np.float32)
    if arr.shape != (12, 1000):
        raise HTTPException(
            status_code=422,
            detail=f"Expected signal shape (12, 1000), got {list(arr.shape)}",
        )
    result = _session.predict(arr)
    report = generate_report(result["predicted_class"], result["confidence"])
    return UploadResponse(
        signal_lead2=arr[1].tolist(),
        report=report,
        **result,
    )
