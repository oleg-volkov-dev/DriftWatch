from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from services.training.train import (FEATURES_BOOL, FEATURES_NUM, LABEL, TrainResult,
                                     build_pipeline, load_csv)

SAMPLE_DATA = {
    "transaction_amount": [
        100.0,
        200.0,
        50.0,
        300.0,
        75.0,
        400.0,
        60.0,
        250.0,
        90.0,
        180.0,
        110.0,
        330.0,
        55.0,
        210.0,
        80.0,
    ],
    "transaction_hour": [10, 14, 2, 20, 8, 22, 11, 15, 3, 18, 9, 21, 4, 16, 7],
    "customer_age": [35, 42, 28, 55, 30, 60, 25, 40, 33, 50, 38, 45, 27, 52, 31],
    "account_tenure_days": [
        365,
        720,
        90,
        1200,
        200,
        900,
        60,
        500,
        150,
        800,
        400,
        1000,
        75,
        650,
        250,
    ],
    "merchant_risk_score": [
        0.1,
        0.5,
        0.8,
        0.3,
        0.2,
        0.7,
        0.4,
        0.6,
        0.9,
        0.15,
        0.25,
        0.65,
        0.85,
        0.35,
        0.12,
    ],
    "geo_distance_km": [
        10.0,
        500.0,
        2.5,
        1200.0,
        45.0,
        800.0,
        5.0,
        300.0,
        1.5,
        600.0,
        20.0,
        700.0,
        3.0,
        400.0,
        50.0,
    ],
    "is_international": [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ],
    "is_fraud": [
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
    ],
}


@pytest.fixture()
def sample_csv(tmp_path: Path) -> str:
    df = pd.DataFrame(SAMPLE_DATA)
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadCsv:
    def test_loads_all_rows(self, sample_csv: str) -> None:
        df = load_csv(sample_csv)
        assert len(df) == 15

    def test_is_international_is_bool(self, sample_csv: str) -> None:
        df = load_csv(sample_csv)
        assert df["is_international"].dtype == bool

    def test_is_fraud_is_bool(self, sample_csv: str) -> None:
        df = load_csv(sample_csv)
        assert df["is_fraud"].dtype == bool

    def test_has_all_feature_columns(self, sample_csv: str) -> None:
        df = load_csv(sample_csv)
        for col in FEATURES_NUM + FEATURES_BOOL + [LABEL]:
            assert col in df.columns


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor_and_classifier(self) -> None:
        pipe = build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "pre" in step_names
        assert "clf" in step_names

    def test_pipeline_fits_and_predicts(self) -> None:
        df = pd.DataFrame(SAMPLE_DATA)
        X = df.drop(columns=[LABEL])
        y = df[LABEL].astype(int)

        pipe = build_pipeline()
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)

        assert probas.shape == (len(df), 2)
        assert (probas >= 0).all() and (probas <= 1).all()


class TestTrainAndLog:
    def test_returns_train_result(self, sample_csv: str) -> None:
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-123"

        with (
            patch("services.training.train.mlflow.set_tracking_uri"),
            patch("services.training.train.mlflow.set_experiment"),
            patch("services.training.train.mlflow.log_metric"),
            patch("services.training.train.mlflow.log_param"),
            patch("services.training.train.mlflow.sklearn.log_model"),
            patch("services.training.train.mlflow.start_run") as mock_start_run,
        ):
            mock_start_run.return_value.__enter__ = lambda s: mock_run
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            from services.training.train import train_and_log

            result = train_and_log(reference_csv=sample_csv)

        assert isinstance(result, TrainResult)
        assert result.run_id == "test-run-id-123"
        assert 0.0 <= result.auc <= 1.0
        assert 0.0 <= result.average_precision <= 1.0

    def test_metrics_are_logged(self, sample_csv: str) -> None:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-456"
        logged_metrics: dict = {}

        def capture_metric(key, value):
            logged_metrics[key] = value

        with (
            patch("services.training.train.mlflow.set_tracking_uri"),
            patch("services.training.train.mlflow.set_experiment"),
            patch("services.training.train.mlflow.log_metric", side_effect=capture_metric),
            patch("services.training.train.mlflow.log_param"),
            patch("services.training.train.mlflow.sklearn.log_model"),
            patch("services.training.train.mlflow.start_run") as mock_start_run,
        ):
            mock_start_run.return_value.__enter__ = lambda s: mock_run
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            from services.training.train import train_and_log

            train_and_log(reference_csv=sample_csv)

        assert "auc" in logged_metrics
        assert "average_precision" in logged_metrics
