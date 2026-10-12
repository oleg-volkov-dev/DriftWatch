"""Label-aware monitoring of one fixed model across two data windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from data.generator.generate import FEATURES

DEFAULT_POLICY = {
    "critical_features": ["transaction_amount", "merchant_risk_score", "is_international"],
    "min_labeled_rows": 100,
    "min_hour_labeled_rows": 100,
    "max_metric_drop": {"precision": 0.10, "recall": 0.10, "auc": 0.05},
    "max_metric_increase": {"brier_score": 0.05, "calibration_error": 0.05},
    "max_fraud_rate_change": 0.10,
}


def monitoring_policy(policy: dict) -> dict:
    configured = policy.get("monitoring_policy", {})
    result = {**DEFAULT_POLICY, **configured}
    if set(configured) - set(DEFAULT_POLICY):
        raise ValueError("Unknown monitoring_policy option")
    critical = result["critical_features"]
    if not isinstance(critical, list) or set(critical) - set(FEATURES):
        raise ValueError("critical_features must be a list of input feature names")
    for key in ("min_labeled_rows", "min_hour_labeled_rows"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, allowed in (
        ("max_metric_drop", {"precision", "recall", "auc", "average_precision"}),
        ("max_metric_increase", {"brier_score", "calibration_error", "log_loss"}),
    ):
        limits = result[key]
        if not isinstance(limits, dict) or set(limits) - allowed:
            raise ValueError(f"Invalid {key} metrics")
        for value in limits.values():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} thresholds must be finite and nonnegative")
    change = result["max_fraud_rate_change"]
    if not np.isfinite(change) or not 0 <= change <= 1:
        raise ValueError("max_fraud_rate_change must be between 0 and 1")
    return result


def labels(data: pd.DataFrame) -> pd.Series:
    raw = data.get("is_fraud", pd.Series(np.nan, index=data.index))
    parsed = raw.astype(str).str.strip().str.lower().map({
        "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0,
    })
    if (raw.notna() & parsed.isna()).any():
        raise ValueError("Monitoring labels must be booleans, 0/1, or missing")
    return parsed


def _metrics(y: np.ndarray, probability: np.ndarray) -> tuple[dict, list]:
    predicted = probability >= 0.5  # Same operating threshold as the inference API.
    positive = y == 1
    tp = int((predicted & positive).sum())
    both_classes = len(np.unique(y)) == 2
    bins = []
    calibration_error = 0.0
    # Include exact 0 and 1, and give each score exactly one bin.
    membership = np.minimum((probability * 10).astype(int), 9)
    for index in range(10):
        selected = membership == index
        count = int(selected.sum())
        if not count:
            continue
        mean_score = float(probability[selected].mean())
        fraction_positive = float(y[selected].mean())
        calibration_error += count / len(y) * abs(mean_score - fraction_positive)
        bins.append({
            "lower": index / 10, "upper": (index + 1) / 10, "count": count,
            "mean_probability": mean_score, "fraud_rate": fraction_positive,
        })
    return {
        "precision": tp / int(predicted.sum()) if predicted.any() else None,
        "recall": tp / int(positive.sum()) if positive.any() else None,
        "auc": float(roc_auc_score(y, probability)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y, probability)) if both_classes else None
        ),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_error": float(calibration_error),
    }, bins


def evaluate_window(data: pd.DataFrame, model, policy: dict) -> dict:
    target = labels(data)
    known = target.notna()
    y = target[known].to_numpy(dtype=int)
    hours = data.loc[known, "transaction_hour"].to_numpy()
    probability = None
    if model is not None and known.any():
        classes = list(model.classes_)
        if len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError("Monitoring requires a binary fraud classifier")
        scores = np.asarray(model.predict_proba(data.loc[known, FEATURES]), dtype=float)
        if scores.shape != (len(y), 2) or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid probabilities")
        if (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1):
            raise ValueError("Model returned invalid probabilities")
        probability = scores[:, classes.index(1)]

    def scope(mask: np.ndarray, minimum: int, total_rows: int) -> dict:
        count = int(mask.sum())
        result = {
            "rows": total_rows,
            "labeled_rows": count,
            "label_coverage": count / total_rows if total_rows else 0.0,
            "status": "insufficient_labels" if count < minimum else "available",
            "metrics": {},
        }
        if count < minimum:
            return result
        result["metrics"]["fraud_rate"] = float(y[mask].mean())
        if probability is None:
            result["status"] = "model_unavailable"
        else:
            metrics, bins = _metrics(y[mask], probability[mask])
            result["metrics"].update(metrics)
            result["calibration_bins"] = bins
        return result

    return {
        "can_retrain": bool(len(data) and known.all() and len(np.unique(y)) == 2),
        "overall": scope(np.ones(len(y), dtype=bool), policy["min_labeled_rows"], len(data)),
        "by_hour": {
            str(hour): scope(
                hours == hour, policy["min_hour_labeled_rows"],
                int((data["transaction_hour"] == hour).sum()),
            )
            for hour in range(24)
        },
    }


def assess_quality(reference: pd.DataFrame, current: pd.DataFrame, model, policy: dict) -> dict:
    ref = evaluate_window(reference, model, policy)
    cur = evaluate_window(current, model, policy)
    alerts = []
    for name in ["overall", *[f"hour_{h}" for h in range(24)]]:
        r = ref["overall"] if name == "overall" else ref["by_hour"][name[5:]]
        c = cur["overall"] if name == "overall" else cur["by_hour"][name[5:]]
        for metric, limit, direction in [
            *[(k, v, -1) for k, v in policy["max_metric_drop"].items()],
            *[(k, v, 1) for k, v in policy["max_metric_increase"].items()],
            ("fraud_rate", policy["max_fraud_rate_change"], 0),
        ]:
            before, after = r["metrics"].get(metric), c["metrics"].get(metric)
            if before is None or after is None:
                continue
            change = (after - before) * direction if direction else abs(after - before)
            if change > limit:
                alerts.append({
                    "kind": "outcome_shift" if metric == "fraud_rate" else "model_quality",
                    "scope": name, "metric": metric, "reference": before, "current": after,
                    "change": change, "threshold": limit, "severity": "medium",
                })
    return {"reference": ref, "current": cur, "alerts": alerts}
