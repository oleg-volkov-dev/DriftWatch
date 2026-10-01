from __future__ import annotations

import json
import os
from pathlib import Path

from services.common.logging import configure_logging, get_logger
from services.control_plane.agents.planner import plan
from services.control_plane.agents.release import maybe_promote_latest_if_gates_pass
from services.control_plane.agents.sentinel import run_sentinel
from services.training.train import train_and_log

logger = get_logger(__name__)


def main() -> None:
    configure_logging("control_plane", json_logs=False)
    logger.info("Control plane orchestrator starting")

    policy_path = os.environ.get(
        "POLICY_PATH", str(Path(__file__).parent / "policies/promotion.yaml")
    )
    data_dir = os.environ.get("DATA_DIR", "/app/shared/data")
    report_dir = os.environ.get("REPORT_DIR", "/app/shared/reports")

    events_dir = Path(os.environ.get("EVENTS_DIR", "/app/shared/events"))
    events_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: Sentinel analysis")
    sentinel_report = run_sentinel(report_dir=report_dir)

    logger.info("Phase 2: Planning")
    plan_obj = plan(sentinel_report, policy_path=policy_path)

    logger.info("Saving agent reports", events_dir=str(events_dir))
    (events_dir / "sentinel_report.json").write_text(
        json.dumps(sentinel_report.__dict__, indent=2), encoding="utf-8"
    )
    (events_dir / "execution_plan.json").write_text(
        json.dumps(plan_obj.__dict__, indent=2), encoding="utf-8"
    )

    if plan_obj.action == "retrain_and_evaluate":
        logger.info("Phase 3: Executing retraining")
        current = Path(data_dir) / "current.csv"
        reference = Path(data_dir) / "reference.csv"

        if not reference.exists() and not current.exists():
            logger.error("No training data found", data_dir=data_dir)
            (events_dir / "training_result.json").write_text(
                json.dumps({"error": f"No training data found in {data_dir}"}), encoding="utf-8"
            )
            raise SystemExit(1)

        train_csv = str(current) if current.exists() else str(reference)
        logger.info("Training on dataset", path=train_csv, using_current=current.exists())

        try:
            train_res = train_and_log(reference_csv=train_csv)
        except Exception as e:
            logger.error("Training phase failed", error=str(e))
            (events_dir / "training_result.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )
            raise SystemExit(1)

        (events_dir / "training_result.json").write_text(
            json.dumps(train_res.__dict__, indent=2), encoding="utf-8"
        )

        logger.info("Phase 4: Release evaluation")
        try:
            release_res = maybe_promote_latest_if_gates_pass(
                plan_obj.policy, run_id=train_res.run_id
            )
        except Exception as e:
            logger.error("Release evaluation failed", error=str(e))
            (events_dir / "release_result.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )
            raise SystemExit(1)

        (events_dir / "release_result.json").write_text(
            json.dumps(release_res.__dict__, indent=2), encoding="utf-8"
        )

        if release_res.promoted and release_res.stage == "Production":
            previous_version = release_res.details.get("previous_version")
            if previous_version:
                manifest = {
                    "model_name": os.environ.get("MODEL_NAME", "fraud_detector"),
                    "promoted_version": release_res.details.get("version"),
                    "rollback_to_version": previous_version,
                }
                (events_dir / "rollback_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                logger.info(
                    "Rollback manifest written",
                    promoted_version=manifest["promoted_version"],
                    rollback_to_version=previous_version,
                )
            else:
                (events_dir / "rollback_manifest.json").unlink(missing_ok=True)
                logger.info("No previous Production version — rollback manifest not written")
            logger.info("Control plane cycle complete - model promoted", stage=release_res.stage)
        elif release_res.promoted:
            logger.info("Control plane cycle complete - model promoted", stage=release_res.stage)
        else:
            logger.warning(
                "Control plane cycle complete - promotion blocked",
                reason=release_res.details.get("reason"),
            )
            if release_res.details.get("reason") not in {
                "quality_gates_failed",
                "promotion_disabled_by_policy",
                "already_in_production",
            }:
                raise SystemExit(1)
    else:
        logger.info("Control plane cycle complete - no action taken", action=plan_obj.action)


if __name__ == "__main__":
    main()
