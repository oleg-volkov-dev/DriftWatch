from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from services.common.features import amount_distance_risk, new_account_risk, payment_activity
from services.common.logging import configure_logging, get_logger

logger = get_logger(__name__)


FEATURES_NUM = [
    "transaction_amount",
    "transaction_hour",
    "payment_attempts_1h",
    "account_tenure_days",
    "merchant_risk_score",
    "geo_distance_km",
]
FEATURES_BOOL = ["is_international"]
LABEL = "is_fraud"


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    auc: float
    average_precision: float


def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading training data", path=path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    required = set(FEATURES_NUM + FEATURES_BOOL + [LABEL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing required columns: {missing}")
    if df.empty:
        raise ValueError("Training CSV must not be empty")
    for column in FEATURES_BOOL + [LABEL]:
        values = df[column].astype(str).str.strip().str.lower()
        parsed = values.map(
            {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
        )
        if parsed.isna().any():
            raise ValueError(f"Training column {column} must contain booleans or 0/1")
        df[column] = parsed.astype(bool)
    if not np.isfinite(df[FEATURES_NUM].to_numpy(dtype=float)).all():
        raise ValueError("Training features must contain finite numbers")
    logger.info("Data loaded", rows=len(df), fraud_rate=f"{df['is_fraud'].mean():.1%}")
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                ["merchant_risk_score"],
            ),
            (
                "amount_distance", FunctionTransformer(amount_distance_risk),
                ["transaction_amount", "geo_distance_km"],
            ),
            (
                "activity", FunctionTransformer(payment_activity),
                ["transaction_amount", "payment_attempts_1h"],
            ),
            ("hour", OneHotEncoder(categories=[list(range(24))]), ["transaction_hour"]),
            ("tenure", FunctionTransformer(new_account_risk), ["account_tenure_days"]),
            ("bool", OneHotEncoder(handle_unknown="ignore"), FEATURES_BOOL),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=500, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])


def select_threshold(y, probability):
    """Minimize FP + 100*FN on validation data; costs are illustrative demo units."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    positives = int(np.asarray(y).sum())
    tp = recall[:-1] * positives
    fp = np.divide(tp * (1 - precision[:-1]), precision[:-1],
                   out=np.full_like(tp, np.inf), where=precision[:-1] > 0)
    costs = np.append(fp + 100 * (positives - tp), 100 * positives)
    return float(np.append(thresholds, 1.0)[np.argmin(costs)])


def fit_model(data):
    """60% fit, 20% threshold validation, 20% untouched release evaluation."""
    train_validation, holdout = train_test_split(
        data, test_size=0.2, stratify=data[LABEL], random_state=42,
    )
    train, validation = train_test_split(
        train_validation, test_size=0.25, stratify=train_validation[LABEL], random_state=43,
    )
    pipe = build_pipeline().fit(train.drop(columns=LABEL), train[LABEL].astype(int))
    pipe.decision_threshold_ = select_threshold(
        validation[LABEL], pipe.predict_proba(validation.drop(columns=LABEL))[:, 1],
    )
    return pipe, holdout


def evaluation_metrics(y, proba) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "brier_score": float(brier_score_loss(y, proba)),
    }


def train_and_log(reference_csv: str, compare_production: bool = False) -> TrainResult:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-demo")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    logger.info("Starting training pipeline", experiment=exp_name, model_name=model_name)

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        raise RuntimeError(f"Cannot connect to MLflow at {tracking_uri}: {e}") from e

    df = load_csv(reference_csv)
    # Reject batches too small to split both classes across fit/validation/test.
    if df[LABEL].value_counts().reindex([False, True], fill_value=0).min() < 5:
        raise ValueError("Training needs at least 5 examples of each class")

    production = None
    production_version = "none"
    if compare_production:
        client = MlflowClient(tracking_uri=tracking_uri)
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
        except Exception as exc:
            # The first training run may precede creation of the registered model.
            if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
                raise
            versions = []
        if versions:
            production_version = str(versions[0].version)
            # Pin the version so a concurrent registry change cannot mix models.
            production = mlflow.sklearn.load_model(f"models:/{model_name}/{production_version}")

    with mlflow.start_run() as run:
        logger.info("Training model", run_id=run.info.run_id)
        pipe, holdout = fit_model(df)
        X_test = holdout.drop(columns=LABEL)
        y_test = holdout[LABEL].astype(int)
        mlflow.log_param("decision_threshold", pipe.decision_threshold_)
        mlflow.log_param("threshold_false_negative_cost", 100)
        mlflow.log_metric("evaluation_positives", int(y_test.sum()))
        mlflow.log_metric("evaluation_negatives", int((y_test == 0).sum()))

        logger.info("Evaluating model on test set")
        proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluation_metrics(y_test, proba)
        auc = metrics["auc"]
        ap = metrics["average_precision"]
        flagged = proba >= pipe.decision_threshold_
        true_positives = int((flagged & (y_test == 1)).sum())
        mlflow.log_metric("recall_at_threshold", true_positives / int(y_test.sum()))
        mlflow.log_metric("flagged_fraction", float(flagged.mean()))
        if flagged.any():
            mlflow.log_metric("precision_at_threshold", true_positives / int(flagged.sum()))

        logger.info("Model evaluation complete", auc=f"{auc:.4f}", average_precision=f"{ap:.4f}")

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        if compare_production:
            mlflow.log_param("comparison_production_version", production_version)
            mlflow.log_metric("evaluation_rows", len(X_test))
            if production is not None:
                production_metrics = evaluation_metrics(y_test, production.predict_proba(X_test)[:, 1])
                for name, value in production_metrics.items():
                    mlflow.log_metric(f"production_{name}", value)
                logger.info(
                    "Candidate compared with Production on held-out data",
                    production_version=production_version,
                    candidate_auc=auc,
                    production_auc=production_metrics["auc"],
                    candidate_brier_score=metrics["brier_score"],
                    production_brier_score=production_metrics["brier_score"],
                )
        mlflow.log_param("model_type", "logreg")
        mlflow.log_param("features_num", ",".join(FEATURES_NUM))
        mlflow.log_param("features_bool", ",".join(FEATURES_BOOL))

        logger.info("Registering model to MLflow", model_name=model_name)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info("Training pipeline complete", run_id=run.info.run_id)
        return TrainResult(run_id=run.info.run_id, auc=auc, average_precision=ap)


def promote_latest_to_production() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    logger.info("Promoting latest model to Production", model_name=model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to MLflow at {tracking_uri}: {e}") from e
    if not versions:
        logger.error("No model versions found", model_name=model_name)
        raise RuntimeError(f"No versions found for model '{model_name}'")

    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Transitioning model to Production stage",
        model_name=model_name,
        version=latest.version,
        archive_existing=True,
    )

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to promote model '{model_name}' v{latest.version} to Production: {e}"
        ) from e

    logger.info(
        "Model promoted successfully",
        model_name=model_name,
        version=latest.version,
        stage="Production",
    )


def main() -> None:
    configure_logging("training", json_logs=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="Path to reference CSV")
    args = ap.parse_args()

    res = train_and_log(args.reference)
    print(f"run_id={res.run_id} auc={res.auc:.6f} average_precision={res.average_precision:.6f}")


if __name__ == "__main__":
    main()
