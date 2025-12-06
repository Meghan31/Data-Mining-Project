import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmp.config import AppConfig
from dmp.cli import train_pipeline
from dmp.pipeline.inference import predict_next_move


def test_train_and_predict_smoke():
    # Train quickly on a short time window
    cfg = AppConfig.load(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    cfg.data.start_date = "2022-01-01"
    cfg.data.end_date = "2022-06-30"
    res = train_pipeline(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    out = predict_next_move(res.model_path, ticker=cfg.data.tickers[0], days_lookback=60)
    assert 0.0 <= out["prob_up"] <= 1.0

