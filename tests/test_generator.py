from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generator.generate import FEATURES, _load_config, _sigmoid, generate_df

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
            "drift": {
                "type": "feature",
                "amount_scale": 3.0,
                "distance_scale": 1.0,
                "merchant_risk_shift": 0.0,
            },
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["transaction_amount"].mean() > base_df["transaction_amount"].mean()

    def test_merchant_risk_shift_increases_scores(self) -> None:
        base_df = generate_df(BASE_CFG)

        drift_cfg = {
            **BASE_CFG,
            "drift": {
                "type": "feature",
                "amount_scale": 1.0,
                "distance_scale": 1.0,
                "merchant_risk_shift": 0.3,
            },
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["merchant_risk_score"].mean() > base_df["merchant_risk_score"].mean()

    def test_distance_scale_increases_distances(self) -> None:
        base_df = generate_df(BASE_CFG)

        drift_cfg = {
            **BASE_CFG,
            "drift": {
                "type": "feature",
                "amount_scale": 1.0,
                "distance_scale": 5.0,
                "merchant_risk_shift": 0.0,
            },
        }
        drifted_df = generate_df(drift_cfg)

        assert drifted_df["geo_distance_km"].mean() > base_df["geo_distance_km"].mean()

    def test_amounts_stay_within_clipped_bounds(self) -> None:
        drift_cfg = {
            **BASE_CFG,
            "drift": {
                "type": "feature",
                "amount_scale": 100.0,
                "distance_scale": 1.0,
                "merchant_risk_shift": 0.0,
            },
        }
        df = generate_df(drift_cfg)
        assert (df["transaction_amount"] <= 10000).all()


class TestGenerateDfShock:
    def test_card_testing_reduces_amounts_and_increases_overnight_fraud(self) -> None:
        cfg = {**BASE_CFG, "n_rows": 10000}
        baseline = generate_df(cfg)
        attacked = generate_df(
            {**cfg, "drift": {"type": "shock", "shock_name": "card_testing"}}
        )
        overnight = attacked["transaction_hour"].isin(range(6))
        assert attacked.loc[overnight, "transaction_amount"].median() < 5
        assert (
            attacked.loc[overnight, "is_fraud"].mean()
            > baseline.loc[overnight, "is_fraud"].mean() + 0.3
        )
        pd.testing.assert_frame_equal(
            attacked.loc[~overnight, FEATURES], baseline.loc[~overnight, FEATURES]
        )

    @pytest.mark.parametrize("shock_name", ["black_friday", "card_testing"])
    def test_shock_overrides_control_window_amounts_and_fraud(self, shock_name) -> None:
        baseline = generate_df(BASE_CFG)
        attacked = generate_df({
            **BASE_CFG,
            "drift": {
                "type": "shock",
                "shock_name": shock_name,
                "spike_hours": [12, 13],
                "amount_scale": 1.0,
                "fraud_spike_multiplier": 0.0,
            },
        })
        window = attacked["transaction_hour"].isin([12, 13])
        assert window.any()
        assert not attacked.loc[window, "is_fraud"].any()
        pd.testing.assert_frame_equal(attacked[FEATURES], baseline[FEATURES])
        pd.testing.assert_frame_equal(attacked.loc[~window], baseline.loc[~window])

    def test_card_testing_config_matches_defaults_and_is_reproducible(self) -> None:
        path = Path(__file__).parents[1] / "data/generator/config/shock_card_testing.yaml"
        cfg = _load_config(str(path))
        configured = generate_df(cfg)
        defaults = generate_df({
            **cfg, "drift": {"type": "shock", "shock_name": "card_testing"}
        })
        pd.testing.assert_frame_equal(configured, defaults)
        pd.testing.assert_frame_equal(configured, generate_df(cfg))
        assert len(configured) == cfg["n_rows"]
        assert configured["transaction_amount"].between(1, 15000).all()

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
    def test_unsupported_concept_drift_is_rejected(self) -> None:
        cfg = {
            **BASE_CFG,
            "drift": {"type": "concept", "concept_variant": "night_fraud"},
        }
        with pytest.raises(ValueError, match="Unsupported drift type"):
            generate_df(cfg)


def test_sigmoid_extreme_values_do_not_overflow():
    with np.errstate(over="raise"):
        np.testing.assert_array_equal(_sigmoid(np.array([-1000, 0, 1000])), [0, 0.5, 1])


@pytest.mark.parametrize("rows", [0, -1])
def test_invalid_row_count_has_clear_error(rows):
    with pytest.raises(ValueError, match="n_rows"):
        generate_df({**BASE_CFG, "n_rows": rows})


@pytest.mark.parametrize(
    "drift",
    [
        {"type": "typo"},
        {"type": "shock", "shock_name": "unknown"},
        {"type": "feature", "amount_scale": -1},
        {"type": "feature", "distance_scale": float("nan")},
        {"type": "shock", "shock_name": "black_friday", "international_rate": 1.5},
        {"type": "shock", "shock_name": "black_friday", "spike_hours": [24]},
    ],
)
def test_invalid_drift_configuration_is_rejected(drift):
    with pytest.raises(ValueError):
        generate_df({**BASE_CFG, "drift": drift})
