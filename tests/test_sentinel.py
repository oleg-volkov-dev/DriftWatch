from __future__ import annotations

from pathlib import Path

import pytest

from services.control_plane.agents.sentinel import SentinelReport, run_sentinel


class TestRunSentinelMissingSummary:
    def test_returns_noop_when_no_file(self, tmp_path: Path) -> None:
        result = run_sentinel(report_dir=str(tmp_path))

        assert isinstance(result, SentinelReport)
        assert result.incident_type == "none"
        assert result.severity == "none"
        assert result.recommended_action == "noop"
        assert result.evidence == {"reason": "missing_monitoring_summary"}


class TestRunSentinelWithSummary:
    @pytest.mark.parametrize("severity", ["none", "low"])
    def test_no_action_for_low_severity(
        self, severity: str, monitoring_summary_factory
    ) -> None:
        report_dir = monitoring_summary_factory(severity=severity)

        result = run_sentinel(report_dir=str(report_dir))

        assert result.incident_type == "none"
        assert result.severity == severity
        assert result.recommended_action == "noop"

    @pytest.mark.parametrize("severity", ["medium", "high"])
    def test_drift_incident_for_high_severity(
        self, severity: str, monitoring_summary_factory
    ) -> None:
        report_dir = monitoring_summary_factory(severity=severity)

        result = run_sentinel(report_dir=str(report_dir))

        assert result.incident_type == "drift"
        assert result.severity == severity
        assert result.recommended_action == "retrain_and_evaluate"

    def test_evidence_contains_summary_data(self, monitoring_summary_factory) -> None:
        report_dir = monitoring_summary_factory(severity="medium", drift_ratio=0.4, drifted_features=3)

        result = run_sentinel(report_dir=str(report_dir))

        assert result.evidence["drift_ratio"] == pytest.approx(0.4)
        assert result.evidence["drifted_features"] == 3
        assert result.evidence["severity"] == "medium"

    def test_none_severity_evidence_has_summary(self, monitoring_summary_factory) -> None:
        report_dir = monitoring_summary_factory(severity="none", drift_ratio=0.0, drifted_features=0)

        result = run_sentinel(report_dir=str(report_dir))

        assert "severity" in result.evidence
        assert result.evidence["severity"] == "none"
