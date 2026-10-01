from __future__ import annotations

import math
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


@dataclass(frozen=True)
class RollbackResult:
    rolled_back: bool
    restored_version: Optional[str]
    details: Dict[str, Any]


def _get_current_production_version(model_name: str, client: MlflowClient) -> Optional[str]:
    """Return the version number currently in Production, or None if none exists."""
    versions = client.get_latest_versions(model_name, stages=["Production"])
    return str(versions[0].version) if versions else None


def rollback_to_version(
    rollback_version: str, expected_current_version: Optional[str] = None
) -> RollbackResult:
    """Demote the current Production model and restore rollback_version to Production."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info("Rollback initiated", model_name=model_name, target_version=rollback_version)

    try:
        current_version = _get_current_production_version(model_name, client)
        if expected_current_version is not None and current_version != expected_current_version:
            return RollbackResult(False, None, {"reason": "stale_rollback_manifest"})
        client.transition_model_version_stage(
            name=model_name,
            version=rollback_version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        logger.error(
            "Rollback failed",
            model_name=model_name,
            target_version=rollback_version,
            error=str(e),
        )
        return RollbackResult(False, None, {"reason": "rollback_failed", "error": str(e)})

    logger.info(
        "Rollback successful",
        model_name=model_name,
        restored_version=rollback_version,
        demoted_version=current_version,
    )
    return RollbackResult(
        True,
        rollback_version,
        {
            "model": model_name,
            "restored_version": rollback_version,
            "demoted_version": current_version,
        },
    )


def maybe_promote_latest_if_gates_pass(
    policy: Dict[str, Any], run_id: Optional[str] = None
) -> ReleaseResult:
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
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        if run_id is not None:
            versions = [v for v in versions if v.run_id == run_id]
        if not versions:
            return ReleaseResult(False, None, {"reason": "no_model_versions"})
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
    except Exception as e:
        logger.error("Failed to query MLflow candidate", error=str(e))
        return ReleaseResult(False, None, {"reason": "mlflow_query_failed", "error": str(e)})

    if run.info.experiment_id != exp.experiment_id or run.info.status != "FINISHED":
        return ReleaseResult(False, None, {"reason": "invalid_candidate_run"})
    try:
        auc = float(run.data.metrics["auc"])
        ap = float(run.data.metrics["average_precision"])
    except (KeyError, TypeError, ValueError):
        return ReleaseResult(False, None, {"reason": "missing_or_invalid_metrics"})
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (auc, ap)):
        return ReleaseResult(False, None, {"reason": "missing_or_invalid_metrics"})

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

    previous_production_version = (
        _get_current_production_version(model_name, client)
        if promote_stage == "Production"
        else None
    )
    if previous_production_version == str(latest.version):
        return ReleaseResult(False, None, {"reason": "already_in_production"})

    logger.info(
        "Promoting model",
        model_name=model_name,
        version=latest.version,
        target_stage=promote_stage,
        previous_production_version=previous_production_version,
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
        {
            "model": model_name,
            "version": str(latest.version),
            "auc": auc,
            "average_precision": ap,
            "previous_version": previous_production_version,
        },
    )
