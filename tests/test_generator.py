from __future__ import annotations

import numpy as np
import pytest

from data.generator.generate import FEATURES, FraudLogic, _sigmoid, generate_df

BASE_CFG = {
    "seed": 42,
    "n_rows": 200,
    "drift": {"type": "none"},
    "fraud_logic": {
        "international_weight": 1.2,
        "night_weight": 0.9,
        "high_amount_weight": 1.0,
        "merchant_risk_weight": 1.3,
        "distance_weight": 0.6,
        "noise": 0.20,
    },
}


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-9

    def test_large_positive_approaches_one(self) -> None:
        assert _sigmoid(np.array([100.0]))[0] > 0.999

    def test_large_negative_approaches_zero(self) -> None:
        assert _sigmoid(np.array([-100.0]))[0] < 0.001

    def test_output_range(self) -> None:
        x = np.linspace(-10, 10, 100)
        out = _sigmoid(x)
        assert np.all(out > 0) and np.all(out < 1)


class TestGenerateDfBase:
    def setup_method(self) -> None:
        self.df = generate_df(BASE_CFG)

    def test_returns_correct_shape(self) -> None:
        assert len(self.df) == 200

    def test_has_expected_columns(self) -> None:
        expected_cols = FEATURES + ["is_fraud"]
        for col in expected_cols:
            assert col in self.df.columns

    def test_transaction_amount_positive(self) -> None:
        assert (self.df["transaction_amount"] > 0).all()

    def test_transaction_hour_in_range(self) -> None:
        assert self.df["transaction_hour"].between(0, 23).all()

    def test_customer_age_in_range(self) -> None:
        assert self.df["customer_age"].between(18, 90).all()

    def test_merchant_risk_score_in_range(self) -> None:
        assert self.df["merchant_risk_score"].between(0, 1).all()

    def test_geo_distance_nonnegative(self) -> None:
        assert (self.df["geo_distance_km"] >= 0).all()

    def test_is_international_is_bool(self) -> None:
        assert self.df["is_international"].dtype == bool

    def test_is_fraud_is_bool(self) -> None:
        assert self.df["is_fraud"].dtype == bool

    def test_deterministic_with_same_seed(self) -> None:
        df2 = generate_df(BASE_CFG)
        assert (self.df["transaction_amount"].values == df2["transaction_amount"].values).all()
        assert (self.df["is_fraud"].values == df2["is_fraud"].values).all()

    def test_different_seeds_produce_different_data(self) -> None:
        cfg2 = {**BASE_CFG, "seed": 99}
        df2 = generate_df(cfg2)
        assert not (self.df["transaction_amount"].values == df2["transaction_amount"].values).all()


class TestGenerateDfFeatureDrift:
    def test_amount_scale_increases_amounts(self) -> None:
        base_df = generate_df(BASE_CFG)

        drift_cfg = {
            **BASE_CFG,
            "drift": {"type": "feature", "amount_scale": 3.0, "distance_scale": 1.0, "merchant_risk_shift": 0.0},
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["transaction_amount"].mean() > base_df["transaction_amount"].mean()

    def test_merchant_risk_shift_increases_scores(self) -> None:
        base_df = generate_df(BASE_CFG)

        drift_cfg = {
            **BASE_CFG,
            "drift": {"type": "feature", "amount_scale": 1.0, "distance_scale": 1.0, "merchant_risk_shift": 0.3},
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["merchant_risk_score"].mean() > base_df["merchant_risk_score"].mean()

    def test_distance_scale_increases_distances(self) -> None:
        base_df = generate_df(BASE_CFG)

        drift_cfg = {
            **BASE_CFG,
            "drift": {"type": "feature", "amount_scale": 1.0, "distance_scale": 5.0, "merchant_risk_shift": 0.0},
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["geo_distance_km"].mean() > base_df["geo_distance_km"].mean()

    def test_amounts_stay_within_clipped_bounds(self) -> None:
        drift_cfg = {
            **BASE_CFG,
            "drift": {"type": "feature", "amount_scale": 100.0, "distance_scale": 1.0, "merchant_risk_shift": 0.0},
        }
        df = generate_df(drift_cfg)
        assert (df["transaction_amount"] <= 10000).all()


class TestGenerateDfShock:
    def test_black_friday_spikes_amounts_in_spike_hours(self) -> None:
        cfg = {
            **BASE_CFG,
            "n_rows": 2000,
            "drift": {
                "type": "shock",
                "shock_name": "black_friday",
                "spike_hours": [20, 21, 22, 23],
                "amount_scale": 3.0,
                "fraud_spike_multiplier": 1.2,
            },
        }
        df = generate_df(cfg)

        spike = df[df["transaction_hour"].isin([20, 21, 22, 23])]
        non_spike = df[~df["transaction_hour"].isin([20, 21, 22, 23])]

        assert spike["transaction_amount"].mean() > non_spike["transaction_amount"].mean()

    def test_amounts_stay_within_clipped_bounds_shock(self) -> None:
        cfg = {
            **BASE_CFG,
            "drift": {
                "type": "shock",
                "shock_name": "black_friday",
                "spike_hours": [20, 21, 22, 23],
                "amount_scale": 100.0,
                "fraud_spike_multiplier": 1.2,
            },
        }
        df = generate_df(cfg)
        assert (df["transaction_amount"] <= 15000).all()


class TestGenerateDfConceptDrift:
    def test_concept_drift_generates_valid_data(self) -> None:
        cfg = {
            **BASE_CFG,
            "drift": {"type": "concept", "concept_variant": "night_fraud"},
        }
        df = generate_df(cfg)
        assert len(df) == BASE_CFG["n_rows"]
        assert df["is_fraud"].dtype == bool
