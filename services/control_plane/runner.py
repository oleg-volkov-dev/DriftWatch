from __future__ import annotations

import json
import os
from pathlib import Path

from services.control_plane.agents.planner import plan
from services.control_plane.agents.release import maybe_promote_latest_if_gates_pass
from services.control_plane.agents.sentinel import run_sentinel
from services.training.train import train_and_log


def main() -> None:
    policy_path = "/app/services/control_plane/policies/promotion.yaml"
    data_dir = os.environ.get("DATA_DIR", "/app/shared/data")
    report_dir = os.environ.get("REPORT_DIR", "/app/shared/reports")

    events_dir = Path("/app/shared/events")
    events_dir.mkdir(parents=True, exist_ok=True)

    sentinel_report = run_sentinel(report_dir=report_dir)
    plan_obj = plan(sentinel_report, policy_path=policy_path)

    (events_dir / "sentinel_report.json").write_text(
        json.dumps(sentinel_report.__dict__, indent=2), encoding="utf-8"
    )
    (events_dir / "execution_plan.json").write_text(json.dumps(plan_obj.__dict__, indent=2), encoding="utf-8")

    if plan_obj.action == "retrain_and_evaluate":
        reference = str(Path(data_dir) / "reference.csv")
        train_res = train_and_log(reference_csv=reference)

        (events_dir / "training_result.json").write_text(
            json.dumps(train_res.__dict__, indent=2), encoding="utf-8"
        )

        release_res = maybe_promote_latest_if_gates_pass(plan_obj.policy)
        (events_dir / "release_result.json").write_text(
            json.dumps(release_res.__dict__, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
