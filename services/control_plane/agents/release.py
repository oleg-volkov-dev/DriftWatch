from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReleaseResult:
    promoted: bool
    stage: Optional[str]
    details: Dict[str, Any]


def maybe_promote_latest_if_gates_pass(policy: Dict[str, Any]) -> ReleaseResult:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-demo")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    logger.info("Release agent evaluating latest model", model_name=model_name, experiment=exp_name)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        exp = client.get_experiment_by_name(exp_name)
    except Exception as e:
        logger.error("MLflow unreachable", tracking_uri=tracking_uri, error=str(e))
        return ReleaseResult(False, None, {"reason": "mlflow_unreachable", "error": str(e)})

    if not exp:
        logger.error("Experiment not found", experiment_name=exp_name)
        return ReleaseResult(False, None, {"reason": "experiment_not_found"})

    try:
        runs = client.search_runs(
            [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
        )
    except Exception as e:
        logger.error("Failed to query MLflow runs", error=str(e))
        return ReleaseResult(False, None, {"reason": "mlflow_query_failed", "error": str(e)})

    if not runs:
        logger.error("No training runs found", experiment_id=exp.experiment_id)
        return ReleaseResult(False, None, {"reason": "no_runs"})

    run = runs[0]
    auc = float(run.data.metrics.get("auc", 0.0))
    ap = float(run.data.metrics.get("average_precision", 0.0))

    logger.info(
        "Latest model metrics retrieved",
        run_id=run.info.run_id,
        auc=f"{auc:.4f}",
        average_precision=f"{ap:.4f}",
    )

    gates = policy.get("quality_gates", {})
    min_auc = float(gates.get("min_auc", 0.0))
    min_ap = float(gates.get("min_average_precision", 0.0))

    pass_gates = (auc >= min_auc) and (ap >= min_ap)
    if not pass_gates:
        logger.warning(
            "Quality gates failed - promotion blocked",
            auc=f"{auc:.4f}",
            min_auc=f"{min_auc:.4f}",
            average_precision=f"{ap:.4f}",
            min_average_precision=f"{min_ap:.4f}",
            auc_gap=f"{min_auc - auc:.4f}",
            ap_gap=f"{min_ap - ap:.4f}",
        )
        return ReleaseResult(
            False,
            None,
            {
                "reason": "quality_gates_failed",
                "auc": auc,
                "ap": ap,
                "min_auc": min_auc,
                "min_ap": min_ap,
            },
        )

    rel = policy.get("release_policy", {})
    if not bool(rel.get("promote_if_quality_gates_pass", True)):
        logger.info("Promotion disabled by policy")
        return ReleaseResult(False, None, {"reason": "promotion_disabled_by_policy"})

    promote_stage = str(rel.get("promote_stage", "Staging"))

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.error("No model versions found in registry", model_name=model_name)
        return ReleaseResult(False, None, {"reason": "no_model_versions"})

    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Promoting model",
        model_name=model_name,
        version=latest.version,
        target_stage=promote_stage,
    )

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage=promote_stage,
            archive_existing_versions=True,
        )
    except Exception as e:
        logger.error(
            "Model stage transition failed",
            model_name=model_name,
            version=latest.version,
            target_stage=promote_stage,
            error=str(e),
        )
        return ReleaseResult(False, None, {"reason": "promotion_failed", "error": str(e)})

    logger.info(
        "Model promoted successfully",
        model_name=model_name,
        version=latest.version,
        stage=promote_stage,
        auc=f"{auc:.4f}",
        average_precision=f"{ap:.4f}",
    )

    return ReleaseResult(
        True,
        promote_stage,
        {"model": model_name, "version": latest.version, "auc": auc, "average_precision": ap},
    )
