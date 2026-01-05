from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from services.common.logging import get_logger
from services.control_plane.agents.sentinel import SentinelReport

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExecutionPlan:
    action: str
    notes: str
    policy: Dict[str, Any]


def load_policy(policy_path: str) -> Dict[str, Any]:
    logger.info("Loading policy configuration", policy_path=policy_path)
    return yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))


def plan(report: SentinelReport, policy_path: str) -> ExecutionPlan:
    logger.info("Planner agent analyzing sentinel report", incident_type=report.incident_type, severity=report.severity)

    policy = load_policy(policy_path)
    drift_policy = policy.get("drift_policy", {})

    if report.incident_type == "none":
        action = drift_policy.get("on_none", {}).get("action", "noop")
        logger.info("No incident - no action required", action=action)
        return ExecutionPlan(action=action, notes="No incident.", policy=policy)

    if report.severity == "low":
        action = drift_policy.get("on_low", {}).get("action", "noop")
    elif report.severity == "medium":
        action = drift_policy.get("on_medium", {}).get("action", "retrain_and_evaluate")
    else:
        action = drift_policy.get("on_high", {}).get("action", "retrain_and_evaluate")

    logger.info(
        "Execution plan created",
        action=action,
        incident_type=report.incident_type,
        severity=report.severity,
    )

    notes = f"incident_type={report.incident_type} severity={report.severity}"
    return ExecutionPlan(action=action, notes=notes, policy=policy)
