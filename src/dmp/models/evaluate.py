from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_on_frame(model_path: str | Path, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    X = df[feature_cols]
    y = df["target_up"]
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = float(accuracy_score(y, preds))
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y, preds).tolist()
    report = classification_report(y, preds, zero_division=0)

    return {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
    }


def save_metrics(metrics: dict, path: str | Path):
    path = Path(path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

