from __future__ import annotations

import json
import os
from pathlib import Path

from services.common.logging import configure_logging, get_logger
from services.control_plane.agents.release import rollback_to_version

logger = get_logger(__name__)


def main() -> None:
    configure_logging("rollback", json_logs=False)
    events_dir = Path(os.environ.get("EVENTS_DIR", "/app/shared/events"))
    manifest_path = events_dir / "rollback_manifest.json"

    if not manifest_path.exists():
        logger.error("No rollback manifest found — nothing to roll back", path=str(manifest_path))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollback_version = manifest.get("rollback_to_version")

    if not rollback_version:
        logger.error("rollback_manifest.json is missing rollback_to_version", manifest=manifest)
        raise SystemExit(1)

    logger.info(
        "Starting rollback",
        promoted_version=manifest.get("promoted_version"),
        rollback_to_version=rollback_version,
    )

    model_name = os.environ.get("MODEL_NAME", "fraud_detector")
    if manifest.get("model_name") != model_name or not manifest.get("promoted_version"):
        raise ValueError("Rollback manifest does not match the configured model or lacks a version")
    result = rollback_to_version(
        str(rollback_version), expected_current_version=str(manifest["promoted_version"])
    )

    (events_dir / "rollback_result.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )

    if result.rolled_back:
        manifest_path.unlink()
        logger.info(
            "Rollback complete",
            restored_version=result.restored_version,
            demoted_version=result.details.get("demoted_version"),
        )
    else:
        logger.error("Rollback failed", reason=result.details.get("reason"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
