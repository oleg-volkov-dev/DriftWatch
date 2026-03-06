from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


VALID_TXN = {
    "transaction_amount": 150.0,
    "transaction_hour": 14,
    "customer_age": 35,
    "account_tenure_days": 365,
    "merchant_risk_score": 0.2,
    "geo_distance_km": 10.0,
    "is_international": False,
}


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (simulates cold start without MLflow)."""
    import services.api.main as api_module

    with patch.object(api_module, "_load_model"):
        api_module._model = None
        api_module._model_stage = None

        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c

    api_module._model = None
    api_module._model_stage = None


@pytest.fixture()
def client_with_model():
    """TestClient with a mocked model that returns a fixed probability."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.8])

    import services.api.main as api_module

    with patch.object(api_module, "_load_model"):
        api_module._model = mock_model
        api_module._model_stage = "Production"

        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c

    api_module._model = None
    api_module._model_stage = None


class TestHealthEndpoint:
    def test_health_returns_ok(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_health_includes_model_name(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")
        assert "model" in response.json()


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/metrics")
        assert "text/plain" in response.headers["content-type"]


class TestPredictEndpointNoModel:
    def test_predict_without_model_returns_error(self, client_no_model: TestClient) -> None:
        # When no model is loaded the endpoint returns a 5xx error response
        response = client_no_model.post("/predict", json=VALID_TXN)
        assert response.status_code >= 500

    def test_predict_invalid_hour_returns_422(self, client_no_model: TestClient) -> None:
        bad_txn = {**VALID_TXN, "transaction_hour": 25}
        response = client_no_model.post("/predict", json=bad_txn)
        assert response.status_code == 422

    def test_predict_negative_amount_returns_422(self, client_no_model: TestClient) -> None:
        bad_txn = {**VALID_TXN, "transaction_amount": -10.0}
        response = client_no_model.post("/predict", json=bad_txn)
        assert response.status_code == 422

    def test_predict_invalid_risk_score_returns_422(self, client_no_model: TestClient) -> None:
        bad_txn = {**VALID_TXN, "merchant_risk_score": 1.5}
        response = client_no_model.post("/predict", json=bad_txn)
        assert response.status_code == 422


class TestPredictEndpointWithModel:
    def test_predict_returns_200(self, client_with_model: TestClient) -> None:
        response = client_with_model.post("/predict", json=VALID_TXN)
        assert response.status_code == 200

    def test_predict_response_has_expected_fields(self, client_with_model: TestClient) -> None:
        response = client_with_model.post("/predict", json=VALID_TXN)
        body = response.json()
        assert "fraud_probability" in body
        assert "is_fraud" in body
        assert "model_stage" in body

    def test_predict_high_score_flags_as_fraud(self, client_with_model: TestClient) -> None:
        # model returns 0.8, so is_fraud should be True (threshold 0.5)
        response = client_with_model.post("/predict", json=VALID_TXN)
        body = response.json()
        assert body["is_fraud"] is True
        assert body["fraud_probability"] == pytest.approx(0.8)

    def test_predict_low_score_not_fraud(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2])

        import services.api.main as api_module

        with patch.object(api_module, "_load_model"):
            api_module._model = mock_model
            api_module._model_stage = "Production"

            with TestClient(api_module.app, raise_server_exceptions=False) as c:
                response = c.post("/predict", json=VALID_TXN)

        api_module._model = None
        api_module._model_stage = None

        body = response.json()
        assert body["is_fraud"] is False
        assert body["fraud_probability"] == pytest.approx(0.2)

    def test_predict_model_stage_returned(self, client_with_model: TestClient) -> None:
        response = client_with_model.post("/predict", json=VALID_TXN)
        assert response.json()["model_stage"] == "Production"
