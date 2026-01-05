from __future__ import annotations

import json
import os
from pathlib import Path

from services.common.logging import configure_logging, get_logger
from services.control_plane.agents.planner import plan
from services.control_plane.agents.release import maybe_promote_latest_if_gates_pass
from services.control_plane.agents.sentinel import run_sentinel
from services.training.train import train_and_log

configure_logging("control_plane", json_logs=False)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Control plane orchestrator starting")

    policy_path = "/app/services/control_plane/policies/promotion.yaml"
    data_dir = os.environ.get("DATA_DIR", "/app/shared/data")
    report_dir = os.environ.get("REPORT_DIR", "/app/shared/reports")

    events_dir = Path("/app/shared/events")
    events_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: Sentinel analysis")
    sentinel_report = run_sentinel(report_dir=report_dir)

    logger.info("Phase 2: Planning")
    plan_obj = plan(sentinel_report, policy_path=policy_path)

    logger.info("Saving agent reports", events_dir=str(events_dir))
    (events_dir / "sentinel_report.json").write_text(
        json.dumps(sentinel_report.__dict__, indent=2), encoding="utf-8"
    )
    (events_dir / "execution_plan.json").write_text(json.dumps(plan_obj.__dict__, indent=2), encoding="utf-8")

    if plan_obj.action == "retrain_and_evaluate":
        logger.info("Phase 3: Executing retraining")
        reference = str(Path(data_dir) / "reference.csv")
        train_res = train_and_log(reference_csv=reference)

        (events_dir / "training_result.json").write_text(
            json.dumps(train_res.__dict__, indent=2), encoding="utf-8"
        )

        logger.info("Phase 4: Release evaluation")
        release_res = maybe_promote_latest_if_gates_pass(plan_obj.policy)
        (events_dir / "release_result.json").write_text(
            json.dumps(release_res.__dict__, indent=2), encoding="utf-8"
        )

        if release_res.promoted:
            logger.info("Control plane cycle complete - model promoted", stage=release_res.stage)
        else:
            logger.warning("Control plane cycle complete - promotion blocked", reason=release_res.details.get("reason"))
    else:
        logger.info("Control plane cycle complete - no action taken", action=plan_obj.action)


if __name__ == "__main__":
    main()
