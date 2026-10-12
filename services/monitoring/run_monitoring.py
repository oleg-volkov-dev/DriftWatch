from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from data.generator.generate import FEATURES
from services.common.logging import configure_logging, get_logger
from services.control_plane.agents.planner import load_policy
from services.monitoring.quality import assess_quality, monitoring_policy

logger = get_logger(__name__)


_SEVERITY_NUMERIC = {"none": 0, "low": 1, "medium": 2, "high": 3}


def compute_drift_severity(report_dict: dict) -> tuple[dict, dict]:
    drifted = 0
    total = 0
    per_feature: dict[str, int] = {}

    for m in report_dict.get("metrics", []):
        if m.get("metric") == "DataDriftTable":
            table = m.get("result", {}).get("drift_by_columns", {})
            for feature, v in table.items():
                if feature == "is_fraud":
                    continue
                if feature in per_feature or not isinstance(v.get("drift_detected"), bool):
                    raise ValueError(f"Invalid or duplicate drift result for {feature}")
                total += 1
                detected = 1 if v.get("drift_detected") is True else 0
                per_feature[feature] = detected
                drifted += detected

    if not total:
        raise ValueError("Drift report contains no feature results")
    drift_ratio = drifted / total

    severity = "none"
    if drift_ratio >= 0.50:
        severity = "high"
    elif drift_ratio >= 0.20:
        severity = "medium"
    elif drift_ratio > 0.0:
        severity = "low"

    summary = {
        "drift_ratio": drift_ratio,
        "drifted_features": drifted,
        "total_features_checked": total,
        "severity": severity,
    }
    return summary, per_feature


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the drift denominator fixed and reject incomplete monitoring data."""
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Monitoring data missing required features: {sorted(missing)}")
    features = data[FEATURES]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Monitoring features must contain finite values")
    return features


def load_production_model() -> tuple[object, dict]:
    """Resolve the registry stage once so both windows use the same model version."""
    name = os.environ.get("MODEL_NAME", "fraud_detector")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    try:
        versions = MlflowClient().get_latest_versions(name, stages=["Production"])
        if not versions:
            return None, {"status": "unavailable", "reason": "no_production_model"}
        uri = f"models:/{name}/{versions[0].version}"
        model = mlflow.sklearn.load_model(uri)
        return model, {"status": "available", "uri": uri, "version": str(versions[0].version)}
    except Exception as exc:
        logger.warning("Model quality unavailable", error=str(exc))
        return None, {"status": "unavailable", "reason": "model_load_failed"}


def add_quality_signals(summary, per_feature, reference, current, model, policy):
    quality = assess_quality(reference, current, model, policy)
    alerts = [
        {"kind": "critical_feature", "feature": feature, "severity": "medium"}
        for feature in policy["critical_features"] if per_feature.get(feature) == 1
    ] + quality.pop("alerts")
    return {
        **summary,
        "per_feature": per_feature,
        "quality": quality,
        "alerts": alerts,
        "incident_severity": (
            "medium" if alerts and summary["severity"] in {"none", "low"}
            else summary["severity"]
        ),
    }


def push_drift_metrics(summary: dict, per_feature: dict[str, int], pushgateway_url: str) -> None:
    registry = CollectorRegistry()

    Gauge(
        "monitoring_drift_ratio", "Fraction of features with detected drift", registry=registry
    ).set(summary["drift_ratio"])
    Gauge(
        "monitoring_drifted_features", "Number of features with detected drift", registry=registry
    ).set(summary["drifted_features"])
    Gauge(
        "monitoring_total_features", "Number of features checked for drift", registry=registry
    ).set(summary["total_features_checked"])
    Gauge(
        "monitoring_drift_severity",
        "Drift severity (0=none 1=low 2=medium 3=high)",
        registry=registry,
    ).set(_SEVERITY_NUMERIC[summary["severity"]])

    feature_gauge = Gauge(
        "monitoring_feature_drift",
        "Per-feature drift detected (1=drifted 0=stable)",
        ["feature"],
        registry=registry,
    )
    for feature, value in per_feature.items():
        feature_gauge.labels(feature=feature).set(value)

    Gauge(
        "monitoring_incident_severity", "Combined drift and quality severity", registry=registry
    ).set(_SEVERITY_NUMERIC[summary.get("incident_severity", summary["severity"])])
    Gauge("monitoring_alert_count", "Active batch alerts", registry=registry).set(
        len(summary.get("alerts", []))
    )
    alert_gauge = Gauge(
        "monitoring_alert", "Active batch alert (1=active)",
        ["kind", "scope", "signal"], registry=registry,
    )
    for alert in summary.get("alerts", []):
        alert_gauge.labels(
            alert["kind"], alert.get("scope", "overall"),
            alert.get("metric", alert.get("feature", "")),
        ).set(1)
    quality_gauge = Gauge(
        "monitoring_quality", "Observed labelled-window metric; undefined values omitted",
        ["window", "scope", "metric"], registry=registry,
    )
    available = Gauge(
        "monitoring_quality_available", "Enough labels and an evaluated model (1=yes)",
        ["window", "scope"], registry=registry,
    )
    coverage = Gauge(
        "monitoring_label_coverage", "Fraction of rows with observed labels",
        ["window", "scope"], registry=registry,
    )
    support = Gauge(
        "monitoring_labeled_rows", "Number of observed labels",
        ["window", "scope"], registry=registry,
    )
    for window, results in summary.get("quality", {}).items():
        scopes = {"overall": results["overall"], **{
            f"hour_{hour}": values for hour, values in results["by_hour"].items()
        }}
        for scope, values in scopes.items():
            available.labels(window, scope).set(int(values["status"] == "available"))
            coverage.labels(window, scope).set(values["label_coverage"])
            support.labels(window, scope).set(values["labeled_rows"])
            for metric, value in values["metrics"].items():
                if value is not None:
                    quality_gauge.labels(window, scope, metric).set(value)

    push_to_gateway(pushgateway_url, job="driftwatch_monitoring", registry=registry)


def main() -> None:
    configure_logging("monitoring", json_logs=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--report-dir", default="/app/shared/reports")
    ap.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL", ""))
    ap.add_argument("--policy", default=os.environ.get(
        "POLICY_PATH", str(Path(__file__).parents[1] / "control_plane/policies/promotion.yaml")
    ))
    args = ap.parse_args()
    policy = monitoring_policy(load_policy(args.policy))

    logger.info("Starting drift detection", reference=args.reference, current=args.current)

    if not Path(args.reference).exists():
        logger.error("Reference dataset not found", path=args.reference)
        raise FileNotFoundError(f"Reference dataset not found: {args.reference}")
    if not Path(args.current).exists():
        logger.error("Current dataset not found", path=args.current)
        raise FileNotFoundError(f"Current dataset not found: {args.current}")

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    logger.info("Datasets loaded", reference_rows=len(ref), current_rows=len(cur))

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError(
            f"Datasets must not be empty (reference={len(ref)} rows, current={len(cur)} rows)"
        )

    logger.info("Running Evidently drift analysis")
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=select_features(ref),
        current_data=select_features(cur),
    )

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    rdict = report.as_dict()
    summary, per_feature = compute_drift_severity(rdict)
    model, model_info = load_production_model()
    summary = add_quality_signals(summary, per_feature, ref, cur, model, policy)
    summary["model"] = model_info
    summary["monitoring_policy"] = policy
    summary["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    summary["report_path"] = str(html_path)

    logger.info(
        "Drift detection complete",
        severity=summary["severity"],
        drift_ratio=f"{summary['drift_ratio']:.1%}",
        drifted_features=summary["drifted_features"],
        total_features=summary["total_features_checked"],
        incident_severity=summary["incident_severity"],
        alerts=len(summary["alerts"]),
        quality_status=summary["quality"]["current"]["overall"]["status"],
    )

    json_path = out_dir / "monitoring_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    logger.info("Reports saved", html_report=str(html_path), json_summary=str(json_path))

    if args.pushgateway:
        try:
            push_drift_metrics(summary, per_feature, args.pushgateway)
            logger.info("Drift metrics pushed to Pushgateway", url=args.pushgateway)
        except Exception as e:
            logger.warning("Failed to push drift metrics to Pushgateway", error=str(e))


if __name__ == "__main__":
    main()
