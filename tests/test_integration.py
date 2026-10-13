"""Exercise the ML lifecycle with real installed dependencies and isolated local storage."""

import json
import types

import mlflow
import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not isinstance(mlflow, types.ModuleType), reason="Requires installed MLflow and Evidently"
)


def test_local_model_lifecycle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from data.generator.generate import generate_df
    from services.api import main as api
    from services.control_plane import rollback, runner
    from services.monitoring import run_monitoring
    from services.training.train import promote_latest_to_production, train_and_log

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "integration")
    monkeypatch.setenv("MODEL_NAME", "fraud_detector")
    for name in ("EVENTS_DIR", "DATA_DIR", "REPORT_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(api, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "_model_stage", None)
    monkeypatch.setattr(api, "_model_version", None)

    cfg = yaml.safe_load(open("data/generator/config/base.yaml"))
    cfg["n_rows"] = 1000
    reference = generate_df(cfg)
    reference.to_csv(tmp_path / "reference.csv", index=False)
    cfg["drift"] = {
        "type": "feature",
        "amount_scale": 3,
        "distance_scale": 3,
        "merchant_risk_shift": 0.3,
    }
    generate_df(cfg).to_csv(tmp_path / "current.csv", index=False)
    # Comparison also supports bootstrapping before a registered model exists.
    train_and_log(str(tmp_path / "reference.csv"), compare_production=True)
    promote_latest_to_production()

    monkeypatch.setattr(
        "sys.argv",
        [
            "monitor",
            "--reference",
            str(tmp_path / "reference.csv"),
            "--current",
            str(tmp_path / "current.csv"),
            "--report-dir",
            str(tmp_path),
            "--pushgateway",
            "",
        ],
    )
    run_monitoring.main()
    summary = json.loads((tmp_path / "monitoring_summary.json").read_text())
    assert summary["total_features_checked"] == 7
    assert summary["severity"] in {"medium", "high"}
    assert summary["model"]["status"] == "available"
    assert summary["model"]["version"] == "1"
    assert summary["quality"]["current"]["overall"]["status"] == "available"
    assert summary["quality"]["current"]["overall"]["metrics"]["brier_score"] >= 0
    assert any(a["kind"] == "critical_feature" for a in summary["alerts"])

    # Permissive gates exercise promotion; production gate rejection is tested separately.
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "quality_gates": {"min_auc": 0, "min_average_precision": 0},
                "release_policy": {"promote_stage": "Production"},
            }
        )
    )
    monkeypatch.setenv("POLICY_PATH", str(policy))
    runner.main()
    with TestClient(api.app) as client:
        assert client.get("/health").json()["version"] == "2"
        txn = reference.drop(columns=["is_fraud"]).iloc[0].to_dict()
        assert client.post("/predict", json=txn).status_code == 200
        rollback.main()
        assert client.post("/reload").status_code == 200
        assert client.get("/health").json()["version"] == "1"
    assert not (tmp_path / "rollback_manifest.json").exists()


@pytest.mark.parametrize("scenario", ["card_testing", "label_only"])
def test_quality_alerts_with_real_drift_detection(scenario, tmp_path, promotion_policy_path):
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    from data.generator.generate import FEATURES, generate_df
    from services.control_plane.agents.planner import plan
    from services.control_plane.agents.sentinel import run_sentinel
    from services.monitoring.quality import monitoring_policy
    from services.monitoring.run_monitoring import add_quality_signals, compute_drift_severity
    from services.training.train import build_pipeline

    cfg = yaml.safe_load(open("data/generator/config/base.yaml"))
    reference = generate_df(cfg)
    model = build_pipeline().fit(reference[FEATURES], reference.is_fraud)
    if scenario == "card_testing":
        current = generate_df(yaml.safe_load(open("data/generator/config/shock_card_testing.yaml")))
    else:
        current = reference.copy()
        current["is_fraud"] = ~current["is_fraud"]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[FEATURES], current_data=current[FEATURES])
    summary, features = compute_drift_severity(report.as_dict())
    assert summary["drifted_features"] == (1 if scenario == "card_testing" else 0)
    summary = add_quality_signals(summary, features, reference, current, model, monitoring_policy({}))
    assert summary["incident_severity"] == "medium"
    assert any(a["kind"] == "model_quality" for a in summary["alerts"])
    assert any(a.get("scope", "").startswith("hour_") for a in summary["alerts"])
    (tmp_path / "monitoring_summary.json").write_text(json.dumps(summary, allow_nan=False))
    assert plan(run_sentinel(str(tmp_path)), promotion_policy_path).action == "retrain_and_evaluate"


@pytest.mark.parametrize("scenario,promoted", [
    ("drift_feature", False),
    ("shock_black_friday", True),
    ("shock_card_testing", True),
])
def test_demo_release_with_real_policy(scenario, promoted, tmp_path, monkeypatch):
    from data.generator.generate import generate_df
    from services.control_plane import runner
    from services.monitoring import run_monitoring
    from services.training.train import promote_latest_to_production, train_and_log

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "demo-release")
    monkeypatch.setenv("MODEL_NAME", "fraud_detector")
    for name in ("EVENTS_DIR", "DATA_DIR", "REPORT_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("POLICY_PATH", "services/control_plane/policies/promotion.yaml")
    for name, out in [("base", "reference"), (scenario, "current")]:
        cfg = yaml.safe_load(open(f"data/generator/config/{name}.yaml"))
        generate_df(cfg).to_csv(tmp_path / f"{out}.csv", index=False)
    train_and_log(str(tmp_path / "reference.csv"))
    promote_latest_to_production()
    monkeypatch.setattr("sys.argv", [
        "monitor", "--reference", str(tmp_path / "reference.csv"),
        "--current", str(tmp_path / "current.csv"),
        "--report-dir", str(tmp_path), "--pushgateway", "",
    ])
    run_monitoring.main()
    runner.main()
    result = json.loads((tmp_path / "release_result.json").read_text())
    assert result["promoted"] is promoted
    client = mlflow.tracking.MlflowClient()
    assert str(client.get_latest_versions("fraud_detector", stages=["Production"])[0].version) == (
        "2" if promoted else "1"
    )
    if not promoted:
        assert result["details"]["reason"] == "production_comparison_failed"
    else:
        assert result["details"]["comparison"]["brier_improvement"] >= 0.002
