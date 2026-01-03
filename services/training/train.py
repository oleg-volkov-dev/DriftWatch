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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES_NUM = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
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
    df = pd.read_csv(path)
    df["is_international"] = df["is_international"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("bool", OneHotEncoder(handle_unknown="ignore"), FEATURES_BOOL),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=500, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_log(reference_csv: str) -> TrainResult:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-demo")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    df = load_csv(reference_csv)
    X = df.drop(columns=[LABEL])
    y = df[LABEL].astype(int)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    pipe = build_pipeline()

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, proba))
        ap = float(average_precision_score(y_test, proba))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("average_precision", ap)
        mlflow.log_param("model_type", "logreg")
        mlflow.log_param("features_num", ",".join(FEATURES_NUM))
        mlflow.log_param("features_bool", ",".join(FEATURES_BOOL))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
        )

        return TrainResult(run_id=run.info.run_id, auc=auc, average_precision=ap)


def promote_latest_to_production() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.environ.get("MODEL_NAME", "fraud_detector")

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="Path to reference CSV")
    args = ap.parse_args()

    res = train_and_log(args.reference)
    print(f"run_id={res.run_id} auc={res.auc:.6f} average_precision={res.average_precision:.6f}")


if __name__ == "__main__":
    main()
