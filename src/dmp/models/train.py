from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    model_path: Path
    metrics_path: Path
    metrics: dict


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: str | Path,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 6,
) -> TrainResult:
    X = df[feature_cols]
    y = df["target_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocess numeric features
    numeric_features = feature_cols
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)], remainder="drop"
    )

    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=200)
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_test": int(len(y_test)),
    }

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "model.joblib"
    metrics_path = artifacts / "metrics.json"

    joblib.dump({"pipeline": pipe, "features": feature_cols}, model_path)
    with open(metrics_path, "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    return TrainResult(model_path=model_path, metrics_path=metrics_path, metrics=metrics)

