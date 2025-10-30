import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from dmp.data.yfinance_loader import download_history
from dmp.features.engineering import compute_returns, add_technical_indicators, make_feature_frame


def test_feature_pipeline_smoke():
    df = download_history(["AAPL"], "2021-01-01", "2021-03-01")
    df = compute_returns(df)
    df = add_technical_indicators(df)
    feat = make_feature_frame(df, ["return_1d", "sma_ratio_5", "rsi", "volatility"])
    assert not feat.empty
    assert {"target_up", "ticker", "date"}.issubset(set(feat.columns))

