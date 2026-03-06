from __future__ import annotations

import pytest

from services.monitoring.run_monitoring import compute_drift_severity


def _make_report(drifted_cols: list[str], all_cols: list[str]) -> dict:
    """Build a minimal Evidently-style report dict for testing."""
    drift_by_columns = {
        col: {"drift_detected": col in drifted_cols, "stattest_name": "ks"}
        for col in all_cols
    }
    return {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {"drift_by_columns": drift_by_columns},
            }
        ]
    }


ALL_FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
    "account_tenure_days",
    "merchant_risk_score",
    "geo_distance_km",
    "is_international",
]


class TestComputeDriftSeverityNoMetrics:
    def test_empty_report_returns_none_severity(self) -> None:
        result = compute_drift_severity({})

        assert result["severity"] == "none"
        assert result["drift_ratio"] == 0.0
        assert result["drifted_features"] == 0
        assert result["total_features_checked"] == 0

    def test_no_data_drift_table_metric(self) -> None:
        report = {"metrics": [{"metric": "SomeOtherMetric", "result": {}}]}

        result = compute_drift_severity(report)

        assert result["severity"] == "none"
        assert result["drift_ratio"] == 0.0


class TestComputeDriftSeverityThresholds:
    def test_no_drift_is_none_severity(self) -> None:
        report = _make_report(drifted_cols=[], all_cols=ALL_FEATURES)

        result = compute_drift_severity(report)

        assert result["severity"] == "none"
        assert result["drift_ratio"] == 0.0
        assert result["drifted_features"] == 0

    def test_single_drifted_feature_is_low_severity(self) -> None:
        # 1/7 ≈ 14.3% → low
        report = _make_report(drifted_cols=["transaction_amount"], all_cols=ALL_FEATURES)

        result = compute_drift_severity(report)

        assert result["severity"] == "low"
        assert result["drifted_features"] == 1
        assert result["total_features_checked"] == 7

    def test_two_drifted_features_is_medium_severity(self) -> None:
        # 2/7 ≈ 28.6% → medium (>= 20%)
        report = _make_report(
            drifted_cols=["transaction_amount", "geo_distance_km"], all_cols=ALL_FEATURES
        )

        result = compute_drift_severity(report)

        assert result["severity"] == "medium"
        assert result["drifted_features"] == 2

    def test_majority_drifted_is_high_severity(self) -> None:
        # 5/7 ≈ 71.4% → high (>= 50%)
        drifted = ALL_FEATURES[:5]
        report = _make_report(drifted_cols=drifted, all_cols=ALL_FEATURES)

        result = compute_drift_severity(report)

        assert result["severity"] == "high"
        assert result["drifted_features"] == 5

    def test_all_features_drifted_is_high_severity(self) -> None:
        report = _make_report(drifted_cols=ALL_FEATURES, all_cols=ALL_FEATURES)

        result = compute_drift_severity(report)

        assert result["severity"] == "high"
        assert result["drift_ratio"] == pytest.approx(1.0)

    def test_exactly_50_percent_is_high_severity(self) -> None:
        # Need an even number: 4 features, 2 drifted = 50% → high
        cols = ALL_FEATURES[:4]
        report = _make_report(drifted_cols=cols[:2], all_cols=cols)

        result = compute_drift_severity(report)

        assert result["severity"] == "high"
        assert result["drift_ratio"] == pytest.approx(0.5)

    def test_drift_ratio_calculation_is_correct(self) -> None:
        # 3 out of 7 drifted = 3/7
        drifted = ALL_FEATURES[:3]
        report = _make_report(drifted_cols=drifted, all_cols=ALL_FEATURES)

        result = compute_drift_severity(report)

        assert result["drift_ratio"] == pytest.approx(3 / 7)
        assert result["total_features_checked"] == 7
