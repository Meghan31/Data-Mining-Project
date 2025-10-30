from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from dmp.data.yfinance_loader import download_history
from dmp.features.engineering import add_technical_indicators, compute_returns, make_feature_frame


def load_model(model_path: str | Path):
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle["features"]


def prepare_latest_features(ticker: str, start: str, end: str, feature_cols: List[str]) -> pd.DataFrame:
    df = download_history([ticker], start=start, end=end)
    df = compute_returns(df)
    df = add_technical_indicators(df)
    feat = make_feature_frame(df, feature_cols)
    return feat.tail(1)


def predict_next_move(model_path: str | Path, ticker: str, days_lookback: int = 90) -> Dict:
    pipe, feature_cols = load_model(model_path)
    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=days_lookback)).isoformat()
    feat = prepare_latest_features(ticker, start=start_date, end=end_date, feature_cols=feature_cols)
    proba = pipe.predict_proba(feat[feature_cols])[:, 1][0]
    pred = int(proba >= 0.5)
    label = "UP" if pred == 1 else "DOWN"
    return {"ticker": ticker, "prob_up": float(proba), "pred_label": label, "features": feature_cols}

