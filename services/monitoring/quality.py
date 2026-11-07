"""Label-aware monitoring of one fixed model across two data windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from services.common.features import FEATURES, decision_threshold

DEFAULT_POLICY = {
    "critical_features": ["transaction_amount", "payment_attempts_1h", "merchant_risk_score"],
    "min_labeled_rows": 100,
    "min_class_rows": 20,
    "max_metric_drop": {"precision": 0.10, "recall": 0.10, "auc": 0.05},
    "max_metric_increase": {"brier_score": 0.002, "calibration_error": 0.005},
    "max_fraud_rate_change": 0.002,
}


def monitoring_policy(policy: dict) -> dict:
    configured = policy.get("monitoring_policy", {})
    result = {**DEFAULT_POLICY, **configured}
    if set(configured) - set(DEFAULT_POLICY):
        raise ValueError("Unknown monitoring_policy option")
    critical = result["critical_features"]
    if not isinstance(critical, list) or set(critical) - set(FEATURES):
        raise ValueError("critical_features must be a list of input feature names")
    for key in ("min_labeled_rows", "min_class_rows"):
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


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> tuple[dict, list]:
    predicted = probability >= threshold  # Threshold persisted with this model.
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

    count = len(y)
    positives = int(y.sum())
    result: dict = {
        "rows": len(data), "labeled_rows": count, "positive_rows": positives,
        "label_coverage": count / len(data) if len(data) else 0.0,
        "status": "insufficient_labels", "metrics": {},
    }
    if count >= policy["min_labeled_rows"]:
        result["metrics"]["fraud_rate"] = float(y.mean())
        if probability is None:
            result["status"] = "model_unavailable"
        elif min(positives, count - positives) < policy["min_class_rows"]:
            result["status"] = "insufficient_class_support"
        else:
            result["status"] = "available"
            result["decision_threshold"] = decision_threshold(model)
            metrics, bins = _metrics(y, probability, result["decision_threshold"])
            result["metrics"].update(metrics)
            result["calibration_bins"] = bins
    return {
        "can_retrain": bool(len(data) and known.all() and min(positives, count - positives) >= 5),
        "overall": result,
    }


def assess_quality(reference: pd.DataFrame, current: pd.DataFrame, model, policy: dict) -> dict:
    ref = evaluate_window(reference, model, policy)
    cur = evaluate_window(current, model, policy)
    alerts = []
    r, c = ref["overall"], cur["overall"]
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
                "scope": "overall", "metric": metric, "reference": before, "current": after,
                "change": change, "threshold": limit, "severity": "medium",
            })
    return {"reference": ref, "current": cur, "alerts": alerts}
