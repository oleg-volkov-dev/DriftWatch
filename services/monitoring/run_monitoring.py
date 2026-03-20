from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from services.common.logging import configure_logging, get_logger

configure_logging("monitoring", json_logs=False)
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
                total += 1
                detected = 1 if v.get("drift_detected") is True else 0
                per_feature[feature] = detected
                drifted += detected

    drift_ratio = (drifted / total) if total else 0.0

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


def push_drift_metrics(summary: dict, per_feature: dict[str, int], pushgateway_url: str) -> None:
    registry = CollectorRegistry()

    Gauge("monitoring_drift_ratio", "Fraction of features with detected drift", registry=registry).set(
        summary["drift_ratio"]
    )
    Gauge("monitoring_drifted_features", "Number of features with detected drift", registry=registry).set(
        summary["drifted_features"]
    )
    Gauge("monitoring_drift_severity", "Drift severity (0=none 1=low 2=medium 3=high)", registry=registry).set(
        _SEVERITY_NUMERIC[summary["severity"]]
    )

    feature_gauge = Gauge(
        "monitoring_feature_drift",
        "Per-feature drift detected (1=drifted 0=stable)",
        ["feature"],
        registry=registry,
    )
    for feature, value in per_feature.items():
        feature_gauge.labels(feature=feature).set(value)

    push_to_gateway(pushgateway_url, job="driftwatch_monitoring", registry=registry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--report-dir", default="/app/shared/reports")
    ap.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL", ""))
    args = ap.parse_args()

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
    report.run(reference_data=ref, current_data=cur)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    rdict = report.as_dict()
    summary, per_feature = compute_drift_severity(rdict)
    summary["report_path"] = str(html_path)

    logger.info(
        "Drift detection complete",
        severity=summary["severity"],
        drift_ratio=f"{summary['drift_ratio']:.1%}",
        drifted_features=summary["drifted_features"],
        total_features=summary["total_features_checked"],
    )

    json_path = out_dir / "monitoring_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Reports saved", html_report=str(html_path), json_summary=str(json_path))

    if args.pushgateway:
        try:
            push_drift_metrics(summary, per_feature, args.pushgateway)
            logger.info("Drift metrics pushed to Pushgateway", url=args.pushgateway)
        except Exception as e:
            logger.warning("Failed to push drift metrics to Pushgateway", error=str(e))


if __name__ == "__main__":
    main()
