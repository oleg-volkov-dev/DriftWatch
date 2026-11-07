from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from services.common.features import (
    FEATURES as FEATURES,
)
from services.common.features import (
    amount_distance_risk,
    new_account_risk,
    payment_activity,
)
from services.common.logging import configure_logging, get_logger

logger = get_logger(__name__)


SHOCK_DEFAULTS = {
    "black_friday": {
        "spike_hours": [20, 21, 22, 23],
        "amount_scale": 2.0,
        "fraud_spike_multiplier": 1.2,
    },
    "card_testing": {
        "spike_hours": list(range(24)),
        "amount_scale": 0.05,
        "fraud_spike_multiplier": 1.0,
        "affected_fraction": 0.01,
        "attack_log_odds": 4.0,
    },
}


@dataclass(frozen=True)
class FraudLogic:
    international_weight: float
    night_weight: float
    high_amount_weight: float
    merchant_risk_weight: float
    distance_weight: float
    noise: float
    new_account_weight: float = 0.5
    velocity_weight: float = 4.0
    baseline_fraud_rate: float = 0.002


# Convert any number to value between 0 and 1 (get probability from the score)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Generator configuration must be a mapping")
    return cfg


def baseline_features(rng, n):
    """Independent rows with synthetic history summaries, not a transaction ledger."""
    attempts = rng.poisson(0.5, n)
    burst = rng.random(n) < 0.01
    attempts[burst] = rng.integers(10, 31, burst.sum())
    return pd.DataFrame({
        "transaction_amount": np.clip(rng.lognormal(4.2, 0.6, n), 1, 5000),
        "transaction_hour": rng.integers(0, 24, n),
        "payment_attempts_1h": attempts,
        "account_tenure_days": np.clip(rng.gamma(2, 180, n).round(), 1, 3650).astype(int),
        "merchant_risk_score": np.clip(rng.beta(2, 5, n) + 0.05, 0, 1),
        "geo_distance_km": np.clip(rng.lognormal(3.2, 0.7, n), 0, 2000),
        "is_international": rng.random(n) < 0.12,
    })


def risk_score(data, logic):
    amount_distance = amount_distance_risk(data[["transaction_amount", "geo_distance_km"]])
    activity = payment_activity(data[["transaction_amount", "payment_attempts_1h"]])
    return (
        logic.international_weight * data.is_international.to_numpy()
        + logic.night_weight * ((data.transaction_hour <= 5) | (data.transaction_hour >= 22))
        + logic.high_amount_weight * amount_distance[:, 0]
        + logic.distance_weight * amount_distance[:, 1]
        + logic.merchant_risk_weight * data.merchant_risk_score
        + logic.new_account_weight * new_account_risk(data.account_tenure_days)
        + logic.velocity_weight * activity[:, 0]
    ).to_numpy()


def baseline_intercept(logic):
    """Match expected prevalence on a fixed calibration population (seed and size)."""
    rng = np.random.default_rng(2026)
    scores = risk_score(baseline_features(rng, 50000), logic)
    scores += rng.normal(0, logic.noise, len(scores))
    lo, hi = -float(np.max(scores)) - 40, -float(np.min(scores)) + 40
    for _ in range(45):
        mid = (lo + hi) / 2
        if _sigmoid(scores + mid).mean() < logic.baseline_fraud_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    seed = int(cfg.get("seed", 42))
    n = int(cfg.get("n_rows", 5000))
    if n <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)

    drift = cfg.get("drift", {"type": "none"})
    drift_type = drift.get("type", "none")
    if drift_type not in {"none", "feature", "shock"}:
        raise ValueError(f"Unsupported drift type: {drift_type}")
    is_shock = drift_type == "shock"
    if is_shock:
        shock_name = drift.get("shock_name")
        if shock_name not in SHOCK_DEFAULTS:
            raise ValueError(
                f"Unsupported shock_name; expected one of: {', '.join(SHOCK_DEFAULTS)}"
            )
        drift = {**SHOCK_DEFAULTS[shock_name], **drift}
    for key in ("amount_scale", "distance_scale", "fraud_spike_multiplier"):
        value = float(drift.get(key, 1.0))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not np.isfinite(float(drift.get("merchant_risk_shift", 0.0))):
        raise ValueError("merchant_risk_shift must be finite")
    intl_rate = drift.get("international_rate")
    if intl_rate is not None and not 0 <= float(intl_rate) <= 1:
        raise ValueError("international_rate must be between 0 and 1")
    affected_fraction = float(drift.get("affected_fraction", 1.0))
    if not np.isfinite(affected_fraction) or not 0 <= affected_fraction <= 1:
        raise ValueError("affected_fraction must be between 0 and 1")
    spike_hours = drift.get("spike_hours", [20, 21, 22, 23])
    if any(not isinstance(h, int) or not 0 <= h <= 23 for h in spike_hours):
        raise ValueError("spike_hours must contain integers from 0 to 23")
    logic = FraudLogic(**cfg.get("fraud_logic", {}))
    if not all(np.isfinite(v) for v in logic.__dict__.values()) or logic.noise < 0:
        raise ValueError("Fraud logic must be finite and noise must be nonnegative")
    if not 0 < logic.baseline_fraud_rate < 1:
        raise ValueError("baseline_fraud_rate must be between 0 and 1, exclusively")
    if not np.isfinite(drift.get("attack_log_odds", 0)):
        raise ValueError("attack_log_odds must be finite")
    if logic.new_account_weight < 0:
        raise ValueError("new_account_weight must be nonnegative")

    logger.info(
        "Starting data generation",
        n_rows=n,
        seed=seed,
        drift_type=drift_type,
    )

    data = baseline_features(rng, n)
    transaction_hour = data["transaction_hour"].to_numpy()
    payment_attempts_1h = data["payment_attempts_1h"].to_numpy(copy=True)
    account_tenure_days = data["account_tenure_days"].to_numpy()
    merchant_risk_score = data["merchant_risk_score"].to_numpy(copy=True)
    geo_distance_km = data["geo_distance_km"].to_numpy(copy=True)
    is_international = data["is_international"].to_numpy(copy=True)
    transaction_amount = data["transaction_amount"].to_numpy(copy=True)

    # Feature distribution changes
    # A batch distribution shift; this is not a simulated time series.
    if drift_type == "feature":
        amount_scale = float(drift.get("amount_scale", 1.0))
        distance_scale = float(drift.get("distance_scale", 1.0))
        risk_shift = float(drift.get("merchant_risk_shift", 0.0))

        transaction_amount = np.clip(transaction_amount * amount_scale, 1, 10000)
        geo_distance_km = np.clip(geo_distance_km * distance_scale, 0, 5000)
        merchant_risk_score = np.clip(merchant_risk_score + risk_shift, 0, 1)

        logger.info(
            "Applied feature drift",
            amount_scale=amount_scale,
            distance_scale=distance_scale,
            merchant_risk_shift=risk_shift,
        )

    # Time-window shocks: shopping surges or small card-testing transactions.
    if is_shock:
        is_spike = np.isin(transaction_hour, spike_hours)
        # Keep ordinary traffic in the attack window so hour alone does not
        # identify an attack. A separate RNG preserves unaffected rows/labels.
        shock_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        is_affected = is_spike & (shock_rng.random(n) < affected_fraction)
        amount_scale = float(drift.get("amount_scale", 2.0))
        transaction_amount = transaction_amount * np.where(is_affected, amount_scale, 1.0)
        transaction_amount = np.clip(transaction_amount, 1, 15000)

        if shock_name == "card_testing":
            transaction_amount[is_affected] = np.minimum(transaction_amount[is_affected], 10.0)
            payment_attempts_1h[is_affected] = shock_rng.integers(10, 31, is_affected.sum())

        risk_shift = float(drift.get("merchant_risk_shift", 0.0))
        if risk_shift:
            merchant_risk_score = np.clip(merchant_risk_score + risk_shift, 0, 1)

        intl_rate = drift.get("international_rate")
        if intl_rate is not None:
            is_international = rng.random(size=n) < float(intl_rate)

        logger.info(
            "Applied shock event",
            shock_name=shock_name,
            spike_hours=sorted(spike_hours),
            amount_scale=amount_scale,
            merchant_risk_shift=risk_shift,
            international_rate=intl_rate,
            affected_fraction=affected_fraction,
            affected_transactions=int(is_affected.sum()),
        )

    df = pd.DataFrame({
        "transaction_amount": transaction_amount,
        "transaction_hour": transaction_hour,
        "payment_attempts_1h": payment_attempts_1h,
        "account_tenure_days": account_tenure_days,
        "merchant_risk_score": merchant_risk_score,
        "geo_distance_km": geo_distance_km,
        "is_international": is_international,
    })
    # Calibrate against a fixed independent baseline, never the current batch.
    # Thus a changed population or attack can change the observed fraud rate.
    intercept = baseline_intercept(logic)
    score = risk_score(df, logic) + rng.normal(0, logic.noise, size=n)
    if is_shock and shock_name == "card_testing":
        score += is_affected * float(drift.get("attack_log_odds", 4.0))
    prob = _sigmoid(score + intercept)
    if is_shock:
        prob = np.clip(
            prob * np.where(is_affected, float(drift["fraud_spike_multiplier"]), 1.0), 0, 1,
        )
    is_fraud = rng.random(size=n) < prob
    df["is_fraud"] = is_fraud

    fraud_rate = float(is_fraud.mean())
    logger.info(
        "Data generation complete",
        total_transactions=n,
        fraud_count=int(is_fraud.sum()),
        fraud_rate=f"{fraud_rate:.3%}",
        baseline_intercept=intercept,
    )

    return df


def main() -> None:
    configure_logging("data_generator", json_logs=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logger.info("Loading configuration", config_path=args.config)
    cfg = _load_config(args.config)

    df = generate_df(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(
        "Dataset saved",
        output_path=str(out_path),
        size_mb=f"{out_path.stat().st_size / 1024 / 1024:.2f}",
    )


if __name__ == "__main__":
    main()
