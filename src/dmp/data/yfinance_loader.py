from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf


@dataclass
class YFDownloadResult:
    ticker: str
    df: pd.DataFrame


def download_history(tickers: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for one or more tickers and return a tidy DataFrame
    with columns: [ticker, date, open, high, low, close, adj_close, volume]
    """
    frames = []
    for t in tickers:
        data = yf.download(t, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        # yfinance may return a MultiIndex with level 1 as the ticker
        if hasattr(data, "columns") and isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.droplevel(1, axis=1)
            except Exception:
                pass
        if data.empty:
            continue
        data = data.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data["ticker"] = t
        frames.append(data[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]])

    if not frames:
        raise RuntimeError("No data downloaded from yfinance for the given tickers and dates.")

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["ticker", "date"], inplace=True)
    return df
