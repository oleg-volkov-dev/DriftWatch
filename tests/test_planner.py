from __future__ import annotations

from services.control_plane.agents.planner import ExecutionPlan, load_policy, plan
from services.control_plane.agents.sentinel import SentinelReport


def _make_sentinel_report(incident_type: str, severity: str) -> SentinelReport:
    return SentinelReport(
        incident_type=incident_type,
        severity=severity,
        recommended_action="noop" if incident_type == "none" else "retrain_and_evaluate",
        evidence={},
    )


class TestLoadPolicy:
    def test_loads_real_policy_file(self, promotion_policy_path: str) -> None:
        policy = load_policy(promotion_policy_path)

        assert "quality_gates" in policy
        assert "drift_policy" in policy
        assert "release_policy" in policy

    def test_quality_gates_have_expected_keys(self, promotion_policy_path: str) -> None:
        policy = load_policy(promotion_policy_path)
        gates = policy["quality_gates"]

        assert "min_auc" in gates
        assert "min_average_precision" in gates

    def test_drift_policy_covers_all_severities(self, promotion_policy_path: str) -> None:
        policy = load_policy(promotion_policy_path)
        drift_policy = policy["drift_policy"]

        for severity in ("on_none", "on_low", "on_medium", "on_high"):
            assert severity in drift_policy


class TestPlan:
    def test_no_incident_returns_noop(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("none", "none")

        result = plan(report, policy_path=promotion_policy_path)

        assert isinstance(result, ExecutionPlan)
        assert result.action == "noop"

    def test_low_severity_returns_noop(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("drift", "low")

        result = plan(report, policy_path=promotion_policy_path)

        assert result.action == "noop"

    def test_medium_severity_returns_retrain(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("drift", "medium")

        result = plan(report, policy_path=promotion_policy_path)

        assert result.action == "retrain_and_evaluate"

    def test_high_severity_returns_retrain(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("drift", "high")

        result = plan(report, policy_path=promotion_policy_path)

        assert result.action == "retrain_and_evaluate"

    def test_plan_includes_policy(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("drift", "medium")

        result = plan(report, policy_path=promotion_policy_path)

        assert "quality_gates" in result.policy

    def test_notes_contain_incident_info(self, promotion_policy_path: str) -> None:
        report = _make_sentinel_report("drift", "high")

        result = plan(report, policy_path=promotion_policy_path)

        assert "high" in result.notes
        assert "drift" in result.notes

    def test_custom_policy_overrides(self, tmp_path) -> None:
        policy_path = tmp_path / "custom_policy.yaml"
        policy_path.write_text(
            "drift_policy:\n  on_none:\n    action: noop\n  on_low:\n    action: noop\n"
            "  on_medium:\n    action: noop\n  on_high:\n    action: retrain_and_evaluate\n",
            encoding="utf-8",
        )
        report = _make_sentinel_report("drift", "medium")

        result = plan(report, policy_path=str(policy_path))

        assert result.action == "noop"
