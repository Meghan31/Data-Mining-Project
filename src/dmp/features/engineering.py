from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["ticker", "date"], inplace=True)
    df["return_1d"] = df["close"].groupby(df["ticker"]).pct_change()
    next_close = df["close"].groupby(df["ticker"]).shift(-1)
    df["target_up"] = (next_close > df["close"]).astype(int)
    return df


def simple_moving_average(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    return df.groupby("ticker")[col].transform(lambda s: s.rolling(window).mean())


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def add_technical_indicators(df: pd.DataFrame, sma_windows=(5, 10, 20), rsi_window=14, vol_window=10) -> pd.DataFrame:
    df = df.copy()
    for w in sma_windows:
        df[f"sma_{w}"] = simple_moving_average(df, w)
        df[f"sma_ratio_{w}"] = df["close"] / (df[f"sma_{w}"] + 1e-9)
    df["rsi"] = df.groupby("ticker")["close"].apply(lambda s: rsi(s, window=rsi_window)).reset_index(level=0, drop=True)
    df["volatility"] = df.groupby("ticker")["return_1d"].transform(lambda s: s.rolling(vol_window).std())
    return df


def make_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    return df.dropna(subset=cols + ["target_up"]) [["ticker", "date"] + cols + ["target_up"]]
