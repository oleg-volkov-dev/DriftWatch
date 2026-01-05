from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SentinelReport:
    incident_type: str
    severity: str
    recommended_action: str
    evidence: Dict[str, Any]


def run_sentinel(report_dir: str = "/app/shared/reports") -> SentinelReport:
    logger.info("Sentinel agent starting analysis", report_dir=report_dir)

    summary_path = Path(report_dir) / "monitoring_summary.json"
    if not summary_path.exists():
        logger.warning("Monitoring summary not found", path=str(summary_path))
        return SentinelReport(
            incident_type="none",
            severity="none",
            recommended_action="noop",
            evidence={"reason": "missing_monitoring_summary"},
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    severity = str(summary.get("severity", "none"))

    if severity in ("none", "low"):
        logger.info("No actionable incidents detected", severity=severity)
        return SentinelReport(
            incident_type="none",
            severity=severity,
            recommended_action="noop",
            evidence=summary,
        )

    logger.warning(
        "Incident detected",
        incident_type="drift",
        severity=severity,
        drift_ratio=summary.get("drift_ratio"),
        drifted_features=summary.get("drifted_features"),
        recommended_action="retrain_and_evaluate",
    )

    return SentinelReport(
        incident_type="drift",
        severity=severity,
        recommended_action="retrain_and_evaluate",
        evidence=summary,
    )
