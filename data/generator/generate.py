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

configure_logging("data_generator", json_logs=False)
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    seed = int(cfg.get("seed", 42))
    n = int(cfg.get("n_rows", 5000))
    rng = np.random.default_rng(seed)

    drift = cfg.get("drift", {"type": "none"})
    drift_type = drift.get("type", "none")

    logger.info(
        "Starting data generation",
        n_rows=n,
        seed=seed,
        drift_type=drift_type,
    )

    # Base feature distributions
    transaction_hour = rng.integers(0, 24, size=n)
    customer_age = np.clip(rng.normal(38, 12, size=n).round(), 18, 90).astype(int)
    account_tenure_days = np.clip(rng.gamma(2.0, 180.0, size=n).round(), 1, 3650).astype(int)
    merchant_risk_score = np.clip(rng.beta(2, 5, size=n) + 0.05, 0, 1)
    geo_distance_km = np.clip(rng.lognormal(mean=3.2, sigma=0.7, size=n), 0, 2000)
    is_international = rng.random(size=n) < 0.12
    transaction_amount = np.clip(rng.lognormal(mean=4.2, sigma=0.6, size=n), 1, 5000)

    # Drift: feature distribution changes
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

    # Shock: specific time-window spike
    if drift_type == "shock" and drift.get("shock_name") == "black_friday":
        spike_hours = set(drift.get("spike_hours", [20, 21, 22, 23]))
        is_spike = np.array([h in spike_hours for h in transaction_hour])
        amount_scale = float(drift.get("amount_scale", 2.0))
        transaction_amount = transaction_amount * np.where(is_spike, amount_scale, 1.0)
        transaction_amount = np.clip(transaction_amount, 1, 15000)

        logger.info(
            "Applied shock event",
            shock_name="black_friday",
            spike_hours=sorted(spike_hours),
            amount_scale=amount_scale,
            affected_transactions=int(is_spike.sum()),
        )

    logic_cfg = cfg.get("fraud_logic", {})
    logic = FraudLogic(**logic_cfg)

    # Explainable fraud scoring (+ noise)
    night = (transaction_hour <= 5) | (transaction_hour >= 22)
    high_amount = transaction_amount >= np.quantile(transaction_amount, 0.90)

    score = (
        logic.international_weight * is_international.astype(float)
        + logic.night_weight * night.astype(float)
        + logic.high_amount_weight * high_amount.astype(float)
        + logic.merchant_risk_weight * merchant_risk_score
        + logic.distance_weight * (geo_distance_km / (geo_distance_km.max() + 1e-9))
    )

    # Drift: concept changes are represented by config weights; optional variants reserved for extension
    if drift_type == "concept":
        _ = drift.get("concept_variant", "default")

    score = score + rng.normal(0, logic.noise, size=n)

    prob = _sigmoid(score - 2.0)

    if drift_type == "shock" and drift.get("shock_name") == "black_friday":
        spike_hours = set(drift.get("spike_hours", [20, 21, 22, 23]))
        is_spike = np.array([h in spike_hours for h in transaction_hour])
        prob = np.clip(
            prob * np.where(is_spike, float(drift.get("fraud_spike_multiplier", 1.2)), 1.0), 0, 1
        )

    is_fraud = rng.random(size=n) < prob

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
