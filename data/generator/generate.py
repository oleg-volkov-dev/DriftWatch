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
from services.common.logging import configure_logging, get_logger

logger = get_logger(__name__)


FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
    "account_tenure_days",
    "merchant_risk_score",
    "geo_distance_km",
    "is_international",
]


@dataclass(frozen=True)
class FraudLogic:
    international_weight: float
    night_weight: float
    high_amount_weight: float
    merchant_risk_weight: float
    distance_weight: float
    noise: float


# Convert any number to value between 0 and 1 (get probability from the score)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Generator configuration must be a mapping")
    return cfg


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
    if is_shock and drift.get("shock_name") != "black_friday":
        raise ValueError("Unsupported shock_name; expected black_friday")
    for key in ("amount_scale", "distance_scale", "fraud_spike_multiplier"):
        value = float(drift.get(key, 1.0))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not np.isfinite(float(drift.get("merchant_risk_shift", 0.0))):
        raise ValueError("merchant_risk_shift must be finite")
    intl_rate = drift.get("international_rate")
    if intl_rate is not None and not 0 <= float(intl_rate) <= 1:
        raise ValueError("international_rate must be between 0 and 1")
    spike_hours = drift.get("spike_hours", [20, 21, 22, 23])
    if any(not isinstance(h, int) or not 0 <= h <= 23 for h in spike_hours):
        raise ValueError("spike_hours must contain integers from 0 to 23")
    logic = FraudLogic(**cfg.get("fraud_logic", {}))
    if not all(np.isfinite(v) for v in logic.__dict__.values()) or logic.noise < 0:
        raise ValueError("Fraud logic must be finite and noise must be nonnegative")

    logger.info(
        "Starting data generation",
        n_rows=n,
        seed=seed,
        drift_type=drift_type,
    )

    # Base feature distributions
    transaction_hour = rng.integers(0, 24, size=n)
    customer_age = np.clip(rng.normal(38, 12, size=n).round(), 18, 90).astype(int)
    account_tenure_days = np.clip(rng.gamma(2.0, 180.0, size=n).round(), 1, 3650).astype(
        int
    )  # Skew to the right, most accounts are young, cap 10years
    merchant_risk_score = np.clip(
        rng.beta(2, 5, size=n) + 0.05, 0, 1
    )  # Skew to the left, most merchants are low risk
    geo_distance_km = np.clip(
        rng.lognormal(mean=3.2, sigma=0.7, size=n), 0, 2000
    )  # Actual median distance = e^3.2 ~ 24.5km, cap 2000km
    is_international = rng.random(size=n) < 0.12  # 12% of being an international transaction
    transaction_amount = np.clip(
        rng.lognormal(mean=4.2, sigma=0.6, size=n), 1, 5000
    )  # Skew to the right, most transactions are tens of dollars

    # Feature distribution changes
    # Scaling up/down certain columns, trying to simulate how real data slowly changes over time
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

    # Specific time-window spike — simulates a sudden shock event (e.g. Black Friday)
    if is_shock:
        is_spike = np.isin(transaction_hour, spike_hours)
        amount_scale = float(drift.get("amount_scale", 2.0))
        transaction_amount = transaction_amount * np.where(is_spike, amount_scale, 1.0)
        transaction_amount = np.clip(transaction_amount, 1, 15000)

        risk_shift = float(drift.get("merchant_risk_shift", 0.0))
        if risk_shift:
            merchant_risk_score = np.clip(merchant_risk_score + risk_shift, 0, 1)

        intl_rate = drift.get("international_rate")
        if intl_rate is not None:
            is_international = rng.random(size=n) < float(intl_rate)

        logger.info(
            "Applied shock event",
            shock_name="black_friday",
            spike_hours=sorted(spike_hours),
            amount_scale=amount_scale,
            merchant_risk_shift=risk_shift,
            international_rate=intl_rate,
            affected_transactions=int(is_spike.sum()),
        )

    # Trying to estimate how suspicious the transaction, adding some noise as well.
    night = (transaction_hour <= 5) | (transaction_hour >= 22)
    high_amount = transaction_amount >= np.quantile(transaction_amount, 0.90)

    score = (
        logic.international_weight * is_international.astype(float)
        + logic.night_weight * night.astype(float)
        + logic.high_amount_weight * high_amount.astype(float)
        + logic.merchant_risk_weight * merchant_risk_score
        + logic.distance_weight
        * (geo_distance_km / (geo_distance_km.max() + 1e-9))  # adding 1e-9 to prevent division by 0
    )

    # Gaussian noise, so transactions with identical features won't always get the same label
    score = score + rng.normal(0, logic.noise, size=n)

    prob = _sigmoid(
        score - 2.0
    )  # Note: transaction with 0 risk signals has ~12% fraud probability by default

    if is_shock:
        prob = np.clip(
            prob * np.where(is_spike, float(drift.get("fraud_spike_multiplier", 1.2)), 1.0),
            0,
            1,  # Fraud spike multiplier is +20% by default
        )

    is_fraud = rng.random(size=n) < prob  # Bernoulli trial

    df = pd.DataFrame(
        {
            "transaction_amount": transaction_amount.astype(float),
            "transaction_hour": transaction_hour.astype(int),
            "customer_age": customer_age.astype(int),
            "account_tenure_days": account_tenure_days.astype(int),
            "merchant_risk_score": merchant_risk_score.astype(float),
            "geo_distance_km": geo_distance_km.astype(float),
            "is_international": is_international.astype(bool),
            "is_fraud": is_fraud.astype(bool),
        }
    )

    fraud_rate = float(is_fraud.mean())
    logger.info(
        "Data generation complete",
        total_transactions=n,
        fraud_count=int(is_fraud.sum()),
        fraud_rate=f"{fraud_rate:.1%}",
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
