from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from services.control_plane.agents.release import (
    maybe_promote_latest_if_gates_pass,
    rollback_to_version,
)

POLICY = {
    "quality_gates": {"min_auc": 0.8, "min_average_precision": 0.2},
    "release_policy": {"promote_stage": "Production"},
}


@pytest.fixture()
def registry():
    client = MagicMock()
    client.get_experiment_by_name.return_value = NS(experiment_id="exp")
    client.search_model_versions.return_value = [
        NS(version="2", run_id="candidate"),
        NS(version="3", run_id="other"),
    ]
    client.get_latest_versions.return_value = [NS(version="1")]
    client.get_run.return_value = NS(
        info=NS(run_id="candidate", experiment_id="exp", status="FINISHED"),
        data=NS(metrics={"auc": 0.9, "average_precision": 0.5}),
    )
    with patch("services.control_plane.agents.release.MlflowClient", return_value=client):
        yield client


def test_promotes_only_version_from_requested_run(registry):
    result = maybe_promote_latest_if_gates_pass(POLICY, run_id="candidate")
    assert result.promoted
    assert result.details["version"] == "2"
    assert result.details["previous_version"] == "1"
    registry.get_run.assert_called_once_with("candidate")
    assert registry.transition_model_version_stage.call_args.kwargs["version"] == "2"


def test_default_evaluates_metrics_of_version_being_promoted(registry):
    maybe_promote_latest_if_gates_pass(POLICY)
    registry.get_run.assert_called_once_with("other")
    assert registry.transition_model_version_stage.call_args.kwargs["version"] == "3"


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"auc": float("inf"), "average_precision": 0.5},
        {"auc": float("nan"), "average_precision": 0.5},
        {"auc": 0.7, "average_precision": 0.5},
    ],
)
def test_invalid_or_failing_metrics_block_promotion(registry, metrics):
    registry.get_run.return_value.data.metrics = metrics
    assert not maybe_promote_latest_if_gates_pass(POLICY, "candidate").promoted
    registry.transition_model_version_stage.assert_not_called()


@pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED"])
def test_unfinished_run_cannot_be_promoted(registry, status):
    registry.get_run.return_value.info.status = status
    assert not maybe_promote_latest_if_gates_pass(POLICY, "candidate").promoted
    registry.transition_model_version_stage.assert_not_called()


def test_staging_does_not_create_production_rollback_target(registry):
    policy = {**POLICY, "release_policy": {"promote_stage": "Staging"}}
    result = maybe_promote_latest_if_gates_pass(policy, "candidate")
    assert result.promoted
    assert result.details["previous_version"] is None


def test_missing_candidate_cannot_promote_another_run(registry):
    assert not maybe_promote_latest_if_gates_pass(POLICY, "missing").promoted
    registry.transition_model_version_stage.assert_not_called()


def test_stale_manifest_cannot_rollback_newer_release(registry):
    result = rollback_to_version("2", expected_current_version="3")
    assert result.details["reason"] == "stale_rollback_manifest"
    registry.transition_model_version_stage.assert_not_called()


def test_registry_failure_cannot_be_treated_as_no_production(registry):
    registry.get_latest_versions.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        maybe_promote_latest_if_gates_pass(POLICY, "candidate")
    registry.transition_model_version_stage.assert_not_called()
