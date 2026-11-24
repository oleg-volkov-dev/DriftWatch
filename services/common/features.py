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
ATTEMPTS_SATURATION = 20.0
AMOUNT_SATURATION = 300.0
COMPOUND_FEATURES = [
    "transaction_amount", "transaction_hour", "account_tenure_days",
    "merchant_risk_score", "geo_distance_km", "is_international",
]


def new_account_risk(tenure_days):
    return np.exp(-np.asarray(tenure_days, dtype=float) / NEW_ACCOUNT_DECAY_DAYS)


def amount_distance_risk(values):
    values = np.asarray(values, dtype=float)
    return np.column_stack([
        values[:, 0] >= HIGH_AMOUNT_THRESHOLD,
        np.clip(values[:, 1] / DISTANCE_SCALE_KM, 0, 1),
    ])


def payment_activity(values):
    """Legacy step transform retained so registered models keep their semantics."""
    values = np.asarray(values, dtype=float)
    rapid = values[:, 1] >= RAPID_ATTEMPTS_THRESHOLD
    return np.column_stack([rapid, rapid & (values[:, 0] <= SMALL_PAYMENT_THRESHOLD)])


def smooth_payment_activity(values):
    """Gradual, bounded attempt effect; no switch at the tenth attempt.

    A separate function preserves existing pickles' original feature transform.
    New models and generated labels use the same smoothstep curve from 0 to 20.
    """
    values = np.asarray(values, dtype=float)
    fraction = np.clip(values[:, 1] / ATTEMPTS_SATURATION, 0, 1)
    activity = fraction * fraction * (3 - 2 * fraction)
    return np.column_stack([
        activity, activity * (values[:, 0] <= SMALL_PAYMENT_THRESHOLD),
    ])


def compound_payment_risk(values):
    """Bounded interaction for multiple concurrent synthetic risk signals.

    Merchant risk, new-account risk and IP distance multiply; high amount,
    night-time and international status each double that combined contribution.
    Uses the same amount/distance caps as the individual features.
    """
    values = np.asarray(values, dtype=float)
    amount, hour, tenure, merchant, distance, international = values.T
    night = (hour <= 5) | (hour >= 22)
    combined = (
        np.clip(merchant, 0, 1)
        * new_account_risk(tenure)
        * np.clip(distance / DISTANCE_SCALE_KM, 0, 1)
        * np.where(amount >= HIGH_AMOUNT_THRESHOLD, 1.0, 0.5)
        * np.where(night, 1.0, 0.5)
        * np.where(international, 1.0, 0.5)
    )
    return combined.reshape(-1, 1)


def decision_threshold(model):
    # Older models and simple test estimators may not have a fitted threshold.
    value = float(getattr(model, "decision_threshold_", 0.5))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid model decision threshold")
    return value


# Versioned functions: existing MLflow pickles reference the legacy functions
# above by name. Never reinterpret their fitted coefficients on reload/rollback.
def smoothstep(values):
    fraction = np.clip(np.asarray(values, dtype=float), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def amount_distance_risk_v2(values):
    """Gradual amount/distance effects with zero slope at either cap."""
    values = np.asarray(values, dtype=float)
    return np.column_stack([
        smoothstep(values[:, 0] / AMOUNT_SATURATION),
        smoothstep(values[:, 1] / DISTANCE_SCALE_KM),
    ])


def night_risk_v2(hours):
    """Circular local time: overnight peak, gradual morning/evening shoulders.

    Peak from 23:00 to 05:00, fading to zero at 10:00 and 18:00.
    Midnight is adjacent to 23:00, rather than a separate learned category.
    """
    hours = np.asarray(hours, dtype=float)
    distance = np.abs((hours - 2 + 12) % 24 - 12)
    return 1 - smoothstep((distance - 3) / 5)


def payment_activity_v2(values):
    """Repeated small payments fade gradually between $5 and $30."""
    values = np.asarray(values, dtype=float)
    activity = smoothstep(values[:, 1] / ATTEMPTS_SATURATION)
    small_payment = 1 - smoothstep((values[:, 0] - 5) / 25)
    return np.column_stack([activity, activity * small_payment])


def compound_payment_risk_v2(values):
    """Use the same continuous curves for individual and combined effects."""
    values = np.asarray(values, dtype=float)
    amount, hour, tenure, merchant, distance, international = values.T
    amount_distance = amount_distance_risk_v2(np.column_stack([amount, distance]))
    combined = (
        np.clip(merchant, 0, 1)
        * new_account_risk(tenure)
        * amount_distance[:, 1]
        * (0.5 + 0.5 * amount_distance[:, 0])
        * (0.5 + 0.5 * night_risk_v2(hour))
        * np.where(international, 1.0, 0.5)
    )
    return combined.reshape(-1, 1)
