import json

import numpy as np
import pandas as pd
import pytest

from services.control_plane.agents.planner import plan
from services.control_plane.agents.sentinel import run_sentinel
from services.monitoring.quality import assess_quality, evaluate_window, monitoring_policy
from services.monitoring.run_monitoring import add_quality_signals


class RiskModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, data):
        probability = data["merchant_risk_score"].to_numpy()
        return np.column_stack([1 - probability, probability])


def transactions(hours=(2,), per_hour=200):
    risk = np.tile([0.1, 0.9], len(hours) * per_hour // 2)
    return pd.DataFrame({
        "transaction_amount": 100.0,
        "transaction_hour": np.repeat(hours, per_hour),
        "customer_age": 35,
        "account_tenure_days": 180,
        "merchant_risk_score": risk,
        "geo_distance_km": 30.0,
        "is_international": False,
        "is_fraud": risk > 0.5,
    })


POLICY = monitoring_policy({})
STABLE = {"severity": "none", "drift_ratio": 0, "drifted_features": 0, "total_features_checked": 7}


def test_known_metrics_and_calibration_bins():
    result = evaluate_window(transactions(), RiskModel(), POLICY)["overall"]
    assert result["status"] == "available"
    assert result["metrics"]["precision"] == 1
    assert result["metrics"]["recall"] == 1
    assert result["metrics"]["auc"] == 1
    assert result["metrics"]["brier_score"] == pytest.approx(0.01)
    assert result["metrics"]["calibration_error"] == pytest.approx(0.1)
    assert result["metrics"]["log_loss"] == pytest.approx(-np.log(0.9))
    assert sum(b["count"] for b in result["calibration_bins"]) == 200


def test_stable_model_and_outcomes_do_not_alert():
    result = assess_quality(transactions(), transactions(), RiskModel(), POLICY)
    assert result["alerts"] == []


def test_label_only_change_triggers_control_response(tmp_path, promotion_policy_path):
    reference = transactions()
    current = reference.copy()
    current["is_fraud"] = ~current["is_fraud"]
    summary = add_quality_signals(STABLE, {}, reference, current, RiskModel(), POLICY)
    assert summary["severity"] == "none"
    assert summary["drift_ratio"] == 0
    assert summary["incident_severity"] == "medium"
    assert {a["metric"] for a in summary["alerts"]} >= {"recall", "auc", "brier_score"}
    (tmp_path / "monitoring_summary.json").write_text(json.dumps(summary, allow_nan=False))
    report = run_sentinel(str(tmp_path))
    assert report.incident_type == "model_quality"
    assert plan(report, promotion_policy_path).action == "retrain_and_evaluate"


def test_hourly_regression_is_detected_when_overall_change_is_small():
    reference = transactions(hours=range(24))
    current = reference.copy()
    current.loc[current.transaction_hour == 2, "is_fraud"] = (
        ~current.loc[current.transaction_hour == 2, "is_fraud"]
    )
    result = assess_quality(reference, current, RiskModel(), POLICY)
    assert result["alerts"]
    assert {a["scope"] for a in result["alerts"]} == {"hour_2"}


def test_critical_feature_alert_preserves_drift_ratio(tmp_path, promotion_policy_path):
    summary = add_quality_signals(
        {**STABLE, "severity": "low", "drift_ratio": 1 / 7, "drifted_features": 1},
        {"transaction_amount": 1}, transactions(), transactions(), RiskModel(), POLICY,
    )
    assert summary["severity"] == "low"
    assert summary["drift_ratio"] == 1 / 7
    assert summary["alerts"] == [{
        "kind": "critical_feature", "feature": "transaction_amount", "severity": "medium"
    }]
    (tmp_path / "monitoring_summary.json").write_text(json.dumps(summary))
    assert plan(run_sentinel(str(tmp_path)), promotion_policy_path).action == "retrain_and_evaluate"


def test_unimportant_feature_does_not_escalate():
    result = add_quality_signals(
        {**STABLE, "severity": "low"}, {"customer_age": 1},
        transactions(), transactions(), RiskModel(), POLICY,
    )
    assert result["incident_severity"] == "low"
    assert result["alerts"] == []


@pytest.mark.parametrize("missing", ["all", "partial"])
def test_delayed_labels_defer_retraining(tmp_path, promotion_policy_path, missing):
    data = transactions()
    if missing == "all":
        data = data.drop(columns="is_fraud")
    else:
        data["is_fraud"] = data["is_fraud"].astype(object)
        data.loc[:99, "is_fraud"] = None
    summary = add_quality_signals(STABLE, {"transaction_amount": 1}, transactions(), data,
                                  RiskModel(), POLICY)
    quality = summary["quality"]["current"]
    assert quality["can_retrain"] is False
    assert quality["overall"]["label_coverage"] == (0 if missing == "all" else 0.5)
    (tmp_path / "monitoring_summary.json").write_text(json.dumps(summary, allow_nan=False))
    assert plan(run_sentinel(str(tmp_path)), promotion_policy_path).action == "await_labels"


def test_sparse_slices_skip_quality_metrics_and_alerts():
    reference = transactions(per_hour=50)
    current = reference.copy()
    current["is_fraud"] = ~current["is_fraud"]
    result = assess_quality(reference, current, RiskModel(), POLICY)
    assert result["alerts"] == []
    assert result["current"]["overall"]["status"] == "insufficient_labels"
    assert result["current"]["by_hour"]["2"]["metrics"] == {}


def test_outcome_shift_is_available_even_without_model():
    reference = transactions()
    current = reference.copy()
    current["is_fraud"] = True
    result = assess_quality(reference, current, None, POLICY)
    assert result["current"]["overall"]["status"] == "model_unavailable"
    assert result["current"]["overall"]["metrics"] == {"fraud_rate": 1.0}
    assert {a["kind"] for a in result["alerts"]} == {"outcome_shift"}


def test_single_class_and_no_flagged_predictions_are_not_healthy_zeros():
    data = transactions()
    data["is_fraud"] = False
    data["merchant_risk_score"] = 0.0
    result = evaluate_window(data, RiskModel(), POLICY)
    metrics = result["overall"]["metrics"]
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["brier_score"] == 0
    assert result["can_retrain"] is False
    json.dumps(result, allow_nan=False)


def test_model_class_order_is_respected():
    class ReverseModel(RiskModel):
        classes_ = np.array([1, 0])

        def predict_proba(self, data):
            return super().predict_proba(data)[:, ::-1]

    assert evaluate_window(transactions(), ReverseModel(), POLICY)["overall"]["metrics"]["auc"] == 1


@pytest.mark.parametrize("bad_value", ["yes", 2, float("inf")])
def test_invalid_labels_fail_monitoring(bad_value):
    data = transactions()
    data["is_fraud"] = data["is_fraud"].astype(object)
    data.loc[0, "is_fraud"] = bad_value
    with pytest.raises(ValueError, match="labels"):
        evaluate_window(data, RiskModel(), POLICY)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1.1, -0.1])
def test_invalid_probabilities_fail_monitoring(bad_value):
    data = transactions()
    data.loc[0, "merchant_risk_score"] = bad_value
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_window(data, RiskModel(), POLICY)


@pytest.mark.parametrize("config", [
    {"min_labeled_rows": 0}, {"min_hour_labeled_rows": 1.5},
    {"critical_features": ["is_fraud"]}, {"max_metric_drop": {"auc": -1}},
    {"max_metric_increase": {"typo": 0.1}}, {"max_fraud_rate_change": float("nan")},
    {"typo": 1},
])
def test_invalid_monitoring_policy_is_rejected(config):
    with pytest.raises(ValueError):
        monitoring_policy({"monitoring_policy": config})


def test_policy_can_disable_alerts():
    policy = monitoring_policy({"monitoring_policy": {
        "critical_features": [], "max_metric_drop": {}, "max_metric_increase": {},
        "max_fraud_rate_change": 1.0,
    }})
    reference = transactions()
    current = reference.copy()
    current["is_fraud"] = ~current["is_fraud"]
    summary = add_quality_signals(STABLE, {"transaction_amount": 1}, reference, current,
                                  RiskModel(), policy)
    assert summary["alerts"] == []
