from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from services.common.logging import configure_logging, get_logger

configure_logging("monitoring", json_logs=False)
logger = get_logger(__name__)


def compute_drift_severity(report_dict: dict) -> dict:
    drifted = 0
    total = 0

    for m in report_dict.get("metrics", []):
        if m.get("metric") == "DataDriftTable":
            table = m.get("result", {}).get("drift_by_columns", {})
            for _, v in table.items():
                total += 1
                if v.get("drift_detected") is True:
                    drifted += 1

    drift_ratio = (drifted / total) if total else 0.0

    severity = "none"
    if drift_ratio >= 0.50:
        severity = "high"
    elif drift_ratio >= 0.20:
        severity = "medium"
    elif drift_ratio > 0.0:
        severity = "low"

    return {
        "drift_ratio": drift_ratio,
        "drifted_features": drifted,
        "total_features_checked": total,
        "severity": severity,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--report-dir", default="/app/shared/reports")
    args = ap.parse_args()

    logger.info("Starting drift detection", reference=args.reference, current=args.current)

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    logger.info("Datasets loaded", reference_rows=len(ref), current_rows=len(cur))

    logger.info("Running Evidently drift analysis")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    rdict = report.as_dict()
    summary = compute_drift_severity(rdict)
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


if __name__ == "__main__":
    main()
