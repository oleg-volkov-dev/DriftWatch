"""Fixed synthetic risk scales, independent of the batch being scored."""

import numpy as np

HIGH_AMOUNT_THRESHOLD = 150.0
DISTANCE_SCALE_KM = 200.0
NEW_ACCOUNT_DECAY_DAYS = 180.0
SMALL_PAYMENT_THRESHOLD = 10.0


def new_account_risk(tenure_days):
    """Risk starts at one and decays toward zero as the account matures."""
    return np.exp(-np.asarray(tenure_days, dtype=float) / NEW_ACCOUNT_DECAY_DAYS)


def amount_distance_risk(values):
    """Match the generator's bounded signals instead of extrapolating raw values."""
    values = np.asarray(values, dtype=float)
    return np.column_stack([
        values[:, 0] >= HIGH_AMOUNT_THRESHOLD,
        np.clip(values[:, 1] / DISTANCE_SCALE_KM, 0, 1),
    ])


def small_overnight_payment(values):
    """Expose the amount/hour interaction used by the card-testing scenario."""
    values = np.asarray(values, dtype=float)
    return ((values[:, 0] <= SMALL_PAYMENT_THRESHOLD) & (values[:, 1] <= 5)).astype(
        float
    ).reshape(-1, 1)
