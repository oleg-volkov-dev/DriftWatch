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
    train_and_log(str(tmp_path / "reference.csv"))
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
