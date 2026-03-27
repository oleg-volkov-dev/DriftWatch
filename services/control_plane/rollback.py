from __future__ import annotations

import json
import os
from pathlib import Path

from services.common.logging import configure_logging, get_logger
from services.control_plane.agents.release import rollback_to_version

configure_logging("rollback", json_logs=False)
logger = get_logger(__name__)


def main() -> None:
    events_dir = Path(os.environ.get("EVENTS_DIR", "/app/shared/events"))
    manifest_path = events_dir / "rollback_manifest.json"

    if not manifest_path.exists():
        logger.error("No rollback manifest found — nothing to roll back", path=str(manifest_path))
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollback_version = manifest.get("rollback_to_version")

    if not rollback_version:
        logger.error("rollback_manifest.json is missing rollback_to_version", manifest=manifest)
        return

    logger.info(
        "Starting rollback",
        promoted_version=manifest.get("promoted_version"),
        rollback_to_version=rollback_version,
    )

    result = rollback_to_version(str(rollback_version))

    (events_dir / "rollback_result.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )

    if result.rolled_back:
        logger.info(
            "Rollback complete",
            restored_version=result.restored_version,
            demoted_version=result.details.get("demoted_version"),
        )
    else:
        logger.error("Rollback failed", reason=result.details.get("reason"))


if __name__ == "__main__":
    main()
