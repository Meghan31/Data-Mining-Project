from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_price(df: pd.DataFrame, ticker: str, out_dir: str | Path):
    ensure_dir(out_dir)
    sub = df[df["ticker"] == ticker]
    plt.figure(figsize=(10, 4))
    plt.plot(sub["date"], sub["close"], label="Close")
    plt.title(f"{ticker} Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    path = Path(out_dir) / f"{ticker}_close.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_return_dist(df: pd.DataFrame, out_dir: str | Path):
    ensure_dir(out_dir)
    plt.figure(figsize=(6, 4))
    sns.histplot(df["return_1d"].dropna(), bins=50, kde=True)
    plt.title("Daily Return Distribution")
    plt.tight_layout()
    path = Path(out_dir) / "return_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_corr(df: pd.DataFrame, feature_cols: List[str], out_dir: str | Path):
    ensure_dir(out_dir)
    corr = df[feature_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlations")
    plt.tight_layout()
    path = Path(out_dir) / "feature_correlation.png"
    plt.savefig(path)
    plt.close()
    return path

