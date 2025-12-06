from __future__ import annotations

import os
from pathlib import Path
import zipfile
from typing import Optional

import pandas as pd


KAGGLE_DATASET = "jacksoncrow/stock-market-dataset"


def _kaggle_installed() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return False


def download_kaggle_dataset(out_dir: str | Path) -> Optional[Path]:
    """
    Download the Kaggle dataset using Kaggle API credentials if available.
    Returns the path to the extracted CSV folder or None if not available.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _kaggle_installed():
        return None

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return None

    zip_path = out / "kaggle_stock_dataset.zip"
    extract_dir = out / "kaggle"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Use kaggle CLI
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {out} -f stocks.zip --force || true")

    # The dataset contains multiple zipped files; unify handling
    # Find first zip inside out
    zips = list(out.glob("*.zip"))
    if not zips:
        return None
    # Extract all zips
    for z in zips:
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            continue

    return extract_dir


def load_kaggle_merged_csv(extracted_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Load and merge the Kaggle dataset files if present.
    Returns a tidy DataFrame similar to yfinance format.
    """
    path = Path(extracted_dir)
    candidates = list(path.glob("**/*.csv"))
    if not candidates:
        return None

    # Some versions contain individual files per ticker; we combine a few
    frames = []
    for csv in candidates[:20]:  # limit for practicality
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue

        # Normalize column names if possible
        cols = {c.lower(): c for c in df.columns}
        lower = {c.lower(): c for c in df.columns}
        required = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in lower for c in required):
            continue
        # Ensure ticker column exists
        ticker_col = lower.get("ticker") or lower.get("symbol")
        if not ticker_col:
            # Try to infer from filename
            ticker = csv.stem.split("_")[0].upper()
            df["ticker"] = ticker
        else:
            df.rename(columns={ticker_col: "ticker"}, inplace=True)

        df.rename(
            columns={
                lower.get("date", "date"): "date",
                lower.get("open", "open"): "open",
                lower.get("high", "high"): "high",
                lower.get("low", "low"): "low",
                lower.get("close", "close"): "close",
                lower.get("volume", "volume"): "volume",
            },
            inplace=True,
        )
        # Drop NA and keep only necessary columns
        keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
        df = df[keep].dropna()
        frames.append(df)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["ticker", "date"], inplace=True)
    return merged

