import json
from unittest.mock import patch

import pytest

from services.control_plane import rollback, runner
from services.control_plane.agents.release import ReleaseResult, RollbackResult
from services.training.train import TrainResult


@pytest.fixture()
def cycle(tmp_path, monkeypatch):
    for name in ("EVENTS_DIR", "DATA_DIR", "REPORT_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "fraud_detector")
    (tmp_path / "monitoring_summary.json").write_text('{"severity": "medium"}')
    (tmp_path / "current.csv").write_text("placeholder")
    return tmp_path


def test_runner_passes_exact_training_run_to_release(cycle):
    with (
        patch.object(runner, "train_and_log", return_value=TrainResult("mine", 0.9, 0.5)),
        patch.object(
            runner,
            "maybe_promote_latest_if_gates_pass",
            return_value=ReleaseResult(
                True, "Production", {"version": "2", "previous_version": "1"}
            ),
        ) as release,
    ):
        runner.main()
    assert release.call_args.kwargs["run_id"] == "mine"
    manifest = json.loads((cycle / "rollback_manifest.json").read_text())
    assert manifest["rollback_to_version"] == "1"
    assert manifest["promoted_version"] == "2"


def test_runner_training_failure_exits_nonzero(cycle):
    with patch.object(runner, "train_and_log", side_effect=RuntimeError("bad data")):
        with pytest.raises(SystemExit) as error:
            runner.main()
    assert error.value.code == 1
    assert json.loads((cycle / "training_result.json").read_text())["error"] == "bad data"


def test_runner_registry_failure_exits_nonzero(cycle):
    with (
        patch.object(runner, "train_and_log", return_value=TrainResult("mine", 0.9, 0.5)),
        patch.object(
            runner,
            "maybe_promote_latest_if_gates_pass",
            return_value=ReleaseResult(False, None, {"reason": "mlflow_unreachable"}),
        ),
    ):
        with pytest.raises(SystemExit):
            runner.main()


def test_staging_promotion_preserves_production_rollback_manifest(cycle):
    manifest = cycle / "rollback_manifest.json"
    manifest.write_text('{"existing": true}')
    with (
        patch.object(runner, "train_and_log", return_value=TrainResult("mine", 0.9, 0.5)),
        patch.object(
            runner,
            "maybe_promote_latest_if_gates_pass",
            return_value=ReleaseResult(True, "Staging", {"version": "2"}),
        ),
    ):
        runner.main()
    assert manifest.read_text() == '{"existing": true}'


def test_rollback_checks_model_and_consumes_manifest(cycle):
    manifest = cycle / "rollback_manifest.json"
    manifest.write_text(
        json.dumps(
            {"model_name": "fraud_detector", "promoted_version": "2", "rollback_to_version": "1"}
        )
    )
    with patch.object(
        rollback,
        "rollback_to_version",
        return_value=RollbackResult(True, "1", {"demoted_version": "2"}),
    ) as restore:
        rollback.main()
    restore.assert_called_once_with("1", expected_current_version="2")
    assert not manifest.exists()


def test_rollback_rejects_wrong_model(cycle):
    (cycle / "rollback_manifest.json").write_text(
        json.dumps({"model_name": "other", "promoted_version": "2", "rollback_to_version": "1"})
    )
    with patch.object(rollback, "rollback_to_version") as restore:
        with pytest.raises(ValueError, match="configured model"):
            rollback.main()
    restore.assert_not_called()
