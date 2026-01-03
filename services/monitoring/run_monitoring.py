from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


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

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    rdict = report.as_dict()
    summary = compute_drift_severity(rdict)
    summary["report_path"] = str(html_path)

    json_path = out_dir / "monitoring_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
