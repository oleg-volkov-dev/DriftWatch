"""Shared feature definitions for the synthetic online card-payment demo."""

import numpy as np

FEATURES = [
    "transaction_amount", "transaction_hour", "payment_attempts_1h",
    "account_tenure_days", "merchant_risk_score", "geo_distance_km", "is_international",
]
HIGH_AMOUNT_THRESHOLD = 150.0
DISTANCE_SCALE_KM = 200.0
NEW_ACCOUNT_DECAY_DAYS = 180.0
SMALL_PAYMENT_THRESHOLD = 10.0
RAPID_ATTEMPTS_THRESHOLD = 10


def new_account_risk(tenure_days):
    return np.exp(-np.asarray(tenure_days, dtype=float) / NEW_ACCOUNT_DECAY_DAYS)


def amount_distance_risk(values):
    values = np.asarray(values, dtype=float)
    return np.column_stack([
        values[:, 0] >= HIGH_AMOUNT_THRESHOLD,
        np.clip(values[:, 1] / DISTANCE_SCALE_KM, 0, 1),
    ])


def payment_activity(values):
    """Prior attempts from the same payment session/device, including declines."""
    values = np.asarray(values, dtype=float)
    rapid = values[:, 1] >= RAPID_ATTEMPTS_THRESHOLD
    return np.column_stack([rapid, rapid & (values[:, 0] <= SMALL_PAYMENT_THRESHOLD)])


def decision_threshold(model):
    # Older models and simple test estimators may not have a fitted threshold.
    value = float(getattr(model, "decision_threshold_", 0.5))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid model decision threshold")
    return value
