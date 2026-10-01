from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response

from services.common.logging import configure_logging, get_logger

logger = get_logger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "fraud_detector")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

REQUESTS = Counter("api_requests_total", "Total prediction requests")
ERRORS = Counter("api_errors_total", "Total prediction errors")
LATENCY = Histogram("api_request_latency_seconds", "Prediction latency seconds")
FRAUD_PREDICTIONS = Counter("fraud_predictions_total", "Fraud predictions by outcome", ["result"])
FRAUD_SCORE = Histogram(
    "fraud_score",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class Txn(BaseModel):
    transaction_amount: float = Field(..., ge=0, allow_inf_nan=False)
    transaction_hour: int = Field(..., ge=0, le=23)
    customer_age: int = Field(..., ge=0, le=120)
    account_tenure_days: int = Field(..., ge=0)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    geo_distance_km: float = Field(..., ge=0, allow_inf_nan=False)
    is_international: bool


class Pred(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_stage: Optional[str]


_model = None
_model_stage: Optional[str] = None
_model_version: Optional[str] = None


_model_lock = Lock()
_reload_lock = Lock()


def _load_model() -> None:
    global _model, _model_stage, _model_version

    # Serialize reloads, but keep serving the previous model while downloading.
    with _reload_lock:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise RuntimeError("No Production model found. Promote a model before reloading.")
        version = str(versions[0].version)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        with _model_lock:
            _model, _model_stage, _model_version = model, "Production", version
        logger.info("Model loaded successfully", model_name=MODEL_NAME, version=version)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging("api", json_logs=False)
    try:
        _load_model()
    except Exception as e:
        logger.warning("Model unavailable at startup", error=str(e))
    yield


app = FastAPI(title="Fraud Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    with _model_lock:
        return {
            "ok": True,
            "model": MODEL_NAME,
            "stage": _model_stage,
            "version": _model_version,
            "model_loaded": _model is not None,
        }


@app.post("/reload")
def reload():
    """Reload the model from MLflow registry (picks up latest Production model)."""
    try:
        _load_model()
        return {
            "ok": True,
            "model": MODEL_NAME,
            "stage": _model_stage,
            "version": _model_version,
            "message": f"Model reloaded successfully (stage: {_model_stage})",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to reload model: {str(e)}") from e


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=Pred)
def predict(txn: Txn):
    REQUESTS.inc()

    with _model_lock:
        model, stage = _model, _model_stage

    if model is None:
        ERRORS.inc()
        return Response(
            content='{"error":"No model loaded. Train and register a model first."}',
            status_code=503,
            media_type="application/json",
        )

    with LATENCY.time():
        try:
            df = pd.DataFrame([txn.model_dump()])
            proba = float(model.predict_proba(df)[0, 1])
            if not math.isfinite(proba) or not 0.0 <= proba <= 1.0:
                raise ValueError("Model returned an invalid probability")

            result = Pred(
                fraud_probability=proba,
                is_fraud=proba >= 0.5,
                model_stage=stage,
            )

            FRAUD_PREDICTIONS.labels(result="fraud" if result.is_fraud else "legit").inc()
            FRAUD_SCORE.observe(proba)

            logger.debug(
                "Prediction made",
                fraud_probability=f"{proba:.3f}",
                is_fraud=result.is_fraud,
                model_stage=stage,
            )

            return result
        except Exception as e:
            ERRORS.inc()
            return JSONResponse(content={"error": str(e)}, status_code=500)
