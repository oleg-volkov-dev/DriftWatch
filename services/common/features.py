"""Fixed synthetic risk scales, independent of the batch being scored."""

import numpy as np

HIGH_AMOUNT_THRESHOLD = 150.0
DISTANCE_SCALE_KM = 200.0
NEW_ACCOUNT_DECAY_DAYS = 180.0


def new_account_risk(tenure_days):
    """Risk starts at one and decays toward zero as the account matures."""
    return np.exp(-np.asarray(tenure_days, dtype=float) / NEW_ACCOUNT_DECAY_DAYS)
