"""Behavior contracts for people exploring the demo, across seeds/scenarios."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.generator.generate import FraudLogic, generate_df, risk_score
from services.common.features import (
    amount_distance_risk,
    compound_payment_risk,
    smooth_payment_activity,
)
from services.training.train import fit_model

REPORTED = dict(
    transaction_amount=149, transaction_hour=2, payment_attempts_1h=10,
    account_tenure_days=10, merchant_risk_score=0.8, geo_distance_km=60,
    is_international=True,
)


@pytest.fixture(scope="module", params=[
    ("base", 42), ("base", 73), ("base", 123),
    ("drift_feature", 43), ("shock_black_friday", 44), ("shock_card_testing", 45),
])
def trained(request):
    scenario, seed = request.param
    cfg = yaml.safe_load(Path(f"data/generator/config/{scenario}.yaml").read_text())
    model, _ = fit_model(generate_df({**cfg, "seed": seed}))
    return model, FraudLogic(**cfg["fraud_logic"])


def test_reported_dollar_change_is_small_and_increases_risk(trained):
    model, _ = trained
    rows = pd.DataFrame([dict(REPORTED, transaction_amount=a) for a in range(100, 201)])
    differences = np.diff(model.predict_proba(rows)[:, 1])
    assert np.all(differences >= -1e-12)
    assert differences.max() < 0.02  # Less than two percentage points per dollar.


def test_no_numeric_cliffs_in_model_or_simulator(trained):
    model, logic = trained
    for feature, boundaries in {
        "transaction_amount": [5, 10, 30, 150, 300],
        "geo_distance_km": [200],
        "payment_attempts_1h": [10, 20],
        "transaction_hour": [0, 5, 6, 10, 18, 22, 23, 24],
    }.items():
        for boundary in boundaries:
            rows = pd.DataFrame([
                dict(REPORTED, **{feature: boundary + delta}) for delta in [-1e-6, 1e-6]
            ])
            # Fractional hours here test the underlying continuous transform;
            # the public API still accepts only whole hours.
            assert np.ptp(model.predict_proba(rows)[:, 1]) < 1e-5
            assert np.ptp(risk_score(rows, logic)) < 1e-5


def test_direction_and_sensitivity_across_transaction_contexts(trained):
    model, _ = trained
    rng = np.random.default_rng(2026)
    contexts = pd.DataFrame({
        "transaction_amount": rng.uniform(30, 300, 64),
        "transaction_hour": rng.integers(0, 24, 64),
        "payment_attempts_1h": rng.integers(0, 21, 64),
        "account_tenure_days": rng.integers(0, 1000, 64),
        "merchant_risk_score": rng.uniform(0, 0.99, 64),
        "geo_distance_km": rng.uniform(0, 200, 64),
        "is_international": rng.random(64) < 0.5,
    })
    contexts = pd.concat([contexts, pd.DataFrame([REPORTED])], ignore_index=True)
    before = model.predict_proba(contexts)[:, 1]
    for feature, step, direction, max_change in [
        ("transaction_amount", 1, 1, 0.02),
        ("geo_distance_km", 1, 1, 0.03),
        ("merchant_risk_score", 0.01, 1, 0.06),
        ("account_tenure_days", 1, -1, 0.03),
        ("payment_attempts_1h", 1, 1, 0.25),
    ]:
        changed = contexts.copy()
        changed[feature] += step
        difference = model.predict_proba(changed)[:, 1] - before
        assert np.all(direction * difference >= -1e-12), feature
        assert np.max(np.abs(difference)) < max_change, feature
    domestic, international = contexts.copy(), contexts.copy()
    domestic["is_international"] = False
    international["is_international"] = True
    assert np.all(model.predict_proba(international)[:, 1] >= model.predict_proba(domestic)[:, 1])
    hours = pd.DataFrame([dict(REPORTED, transaction_hour=h) for h in range(24)])
    probabilities = model.predict_proba(hours)[:, 1]
    assert probabilities[23] == pytest.approx(probabilities[0])
    assert probabilities[2] >= probabilities[12]
    assert np.abs(np.diff(np.r_[probabilities, probabilities[0]])).max() < 0.15


def test_legacy_functions_keep_saved_model_semantics():
    np.testing.assert_array_equal(amount_distance_risk([[149, 60], [150, 60]]),
                                  [[0, 0.3], [1, 0.3]])
    rows = [[149, 2, 10, 0.8, 60, 1], [150, 2, 10, 0.8, 60, 1]]
    risk = compound_payment_risk(rows)[:, 0]
    assert risk[1] == 2 * risk[0]
    # Old smooth-attempt models still have the original $10 step.
    np.testing.assert_array_equal(smooth_payment_activity([[10, 10], [11, 10]]),
                                  [[0.5, 0.5], [0.5, 0]])
