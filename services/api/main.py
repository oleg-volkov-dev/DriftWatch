from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI
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


def _load_model() -> None:
    global _model, _model_stage

    logger.info(
        "Loading model from MLflow", model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    stage_uri = f"models:/{MODEL_NAME}/Production"
    try:
        _model = mlflow.pyfunc.load_model(stage_uri)
        _model_stage = "Production"
        logger.info("Model loaded successfully", model_name=MODEL_NAME, stage="Production")
        return
    except Exception:
        logger.debug("Production model not found, trying latest")

    latest_uri = f"models:/{MODEL_NAME}/latest"
    try:
        _model = mlflow.pyfunc.load_model(latest_uri)
        _model_stage = "latest"
        logger.info("Model loaded successfully", model_name=MODEL_NAME, stage="latest")
    except Exception:
        _model = None
        _model_stage = "none"
        logger.warning("No model found in MLflow", model_name=MODEL_NAME)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Fraud Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "stage": _model_stage}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=Pred)
def predict(txn: Txn):
    REQUESTS.inc()

    if _model is None:
        ERRORS.inc()
        logger.error("Prediction request rejected - no model loaded")
        return Response(
            content='{"error":"No model loaded. Train and register a model first."}',
            status_code=503,
            media_type="application/json",
        )

    with LATENCY.time():
        try:
            df = pd.DataFrame([txn.model_dump()])
            score = float(_model.predict(df)[0])
            proba = max(0.0, min(1.0, score))

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
            logger.error("Prediction failed", error=str(e), exc_info=True)
            return Response(
                content=f'{{"error":"{str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )
