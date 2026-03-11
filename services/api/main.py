from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from services.common.logging import configure_logging, get_logger

configure_logging("api", json_logs=False)
logger = get_logger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "fraud_detector")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

REQUESTS = Counter("api_requests_total", "Total prediction requests")
ERRORS = Counter("api_errors_total", "Total prediction errors")
LATENCY = Histogram("api_request_latency_seconds", "Prediction latency seconds")


class Txn(BaseModel):
    transaction_amount: float = Field(..., ge=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    customer_age: int = Field(..., ge=0, le=120)
    account_tenure_days: int = Field(..., ge=0)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    geo_distance_km: float = Field(..., ge=0)
    is_international: bool


class Pred(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_stage: Optional[str]


_model = None
_model_stage: Optional[str] = None
_model_version: Optional[str] = None


def _load_model() -> None:
    global _model, _model_stage, _model_version

    logger.info(
        "Loading model from MLflow", model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)

    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            _model_version = versions[0].version
            version_uri = f"models:/{MODEL_NAME}/{_model_version}"
            _model = mlflow.sklearn.load_model(version_uri)
            _model_stage = "Production"
            logger.info("Model loaded successfully", model_name=MODEL_NAME, stage="Production", version=_model_version)
            return
        logger.warning("No Production model found, trying latest", model_name=MODEL_NAME)
    except Exception as e:
        logger.warning("Production model failed to load, trying latest", error=str(e))

    try:
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if all_versions:
            latest = max(all_versions, key=lambda v: int(v.version))
            _model_version = latest.version
            version_uri = f"models:/{MODEL_NAME}/{_model_version}"
            _model = mlflow.sklearn.load_model(version_uri)
            _model_stage = latest.current_stage
            logger.info("Model loaded successfully", model_name=MODEL_NAME, stage=_model_stage, version=_model_version)
        else:
            raise RuntimeError("No model versions found")
    except Exception as e:
        _model = None
        _model_stage = "none"
        _model_version = None
        logger.warning("No model found in MLflow", model_name=MODEL_NAME, error=str(e))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Fraud Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "stage": _model_stage, "version": _model_version, "model_loaded": _model is not None}


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
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=Pred)
def predict(txn: Txn):
    REQUESTS.inc()

    if _model is None:
        ERRORS.inc()
        return Response(
            content='{"error":"No model loaded. Train and register a model first."}',
            status_code=503,
            media_type="application/json",
        )

    with LATENCY.time():
        try:
            df = pd.DataFrame([txn.model_dump()])
            proba = max(0.0, min(1.0, float(_model.predict_proba(df)[0, 1])))

            result = Pred(
                fraud_probability=proba,
                is_fraud=proba >= 0.5,
                model_stage=_model_stage,
            )

            logger.debug(
                "Prediction made",
                fraud_probability=f"{proba:.3f}",
                is_fraud=result.is_fraud,
                model_stage=_model_stage,
            )

            return result
        except Exception as e:
            ERRORS.inc()
            return Response(
                content=f'{{"error":"{str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )
