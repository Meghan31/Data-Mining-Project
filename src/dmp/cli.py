from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import List

import pandas as pd

from dmp.config import AppConfig
from dmp.user_config import load_user_env, apply_overrides
from dmp.data.yfinance_loader import download_history
from dmp.features.engineering import add_technical_indicators, compute_returns, make_feature_frame
from dmp.models.train import train_classifier
from dmp.models.evaluate import evaluate_on_frame, save_metrics
from dmp.pipeline.inference import predict_next_move
from dmp.visualization.eda import plot_corr, plot_price, plot_return_dist
from dmp.apis.finnhub_client import FinnhubClient
from dmp.apis.llm import analyze_with_vader, summarize_openai


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    # Load user env first to expose keys to libraries
    load_user_env(Path(path).with_name("user_config.env"))
    cfg = AppConfig.load(path)
    cfg = apply_overrides(cfg)
    return cfg


def prepare_dataframe(cfg: AppConfig) -> pd.DataFrame:
    if cfg.data.source == "yfinance":
        df = download_history(cfg.data.tickers, cfg.data.start_date, cfg.data.end_date, cfg.data.interval)
    else:
        # For this implementation we prefer yfinance; Kaggle path optional
        df = download_history(cfg.data.tickers, cfg.data.start_date, cfg.data.end_date, cfg.data.interval)
    df = compute_returns(df)
    df = add_technical_indicators(
        df, sma_windows=cfg.features.sma_windows, rsi_window=cfg.features.rsi_window, vol_window=cfg.features.volatility_window
    )
    return df


def train_pipeline(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    df = prepare_dataframe(cfg)
    feature_cols = [
        "return_1d",
        *[f"sma_ratio_{w}" for w in cfg.features.sma_windows],
        "rsi",
        "volatility",
    ]
    feat = make_feature_frame(df, feature_cols)
    res = train_classifier(
        feat,
        feature_cols=feature_cols,
        artifacts_dir=cfg.general.artifacts_dir,
        model_type=cfg.model.type,
        test_size=cfg.model.test_size,
        random_state=cfg.general.random_state,
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
    )
    print("Metrics:", json.dumps(res.metrics, indent=2))
    return res


def run_eda(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    df = prepare_dataframe(cfg)
    figs = []
    for t in cfg.data.tickers[:3]:
        figs.append(str(plot_price(df, t, cfg.general.figures_dir)))
    feature_cols = ["return_1d", *[f"sma_ratio_{w}" for w in cfg.features.sma_windows], "rsi", "volatility"]
    df_feat = make_feature_frame(df, feature_cols)
    figs.append(str(plot_return_dist(df_feat, cfg.general.figures_dir)))
    figs.append(str(plot_corr(df_feat[feature_cols], feature_cols, cfg.general.figures_dir)))
    print("Saved EDA figures:")
    for f in figs:
        print(" -", f)


def evaluate(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    df = prepare_dataframe(cfg)
    feature_cols = [
        "return_1d",
        *[f"sma_ratio_{w}" for w in cfg.features.sma_windows],
        "rsi",
        "volatility",
    ]
    feat = make_feature_frame(df, feature_cols)
    metrics = evaluate_on_frame(Path(cfg.general.artifacts_dir) / "model.joblib", feat, feature_cols)
    save_metrics(metrics, Path(cfg.general.artifacts_dir) / "eval_metrics.json")
    print(json.dumps(metrics, indent=2))


def live_inference(config_path: str | Path = "config.yaml", ticker: str | None = None):
    cfg = load_config(config_path)
    ticker = ticker or cfg.live.default_ticker
    res = predict_next_move(Path(cfg.general.artifacts_dir) / "model.joblib", ticker)
    print("Model inference:", json.dumps(res, indent=2))

    # Fetch news via Finnhub (if key present)
    import os
    api_key = os.getenv("FINNHUB_API_KEY")
    headlines: List[str] = []
    if api_key:
        client = FinnhubClient(api_key)
        today = dt.date.today()
        start = (today - dt.timedelta(days=7)).isoformat()
        news = client.company_news(ticker, start, today.isoformat(), max_items=cfg.llm.max_news)
        headlines = [n.get("headline", "") for n in news]
    else:
        headlines = [
            f"{ticker} product update gains traction among users",
            f"Analyst revises {ticker} target price",
        ]

    llm_summary = summarize_openai(headlines, model=cfg.llm.model)
    if not llm_summary and cfg.llm.use_vader_fallback:
        score, label = analyze_with_vader(headlines)
        llm_summary = {"sentiment_score": score, "label": label, "summary": "VADER fallback analysis"}

    # Aggregate
    sentiment = llm_summary.get("sentiment_score", 0.5) if llm_summary else 0.5
    model_prob = res["prob_up"]
    final_score = 0.6 * model_prob + 0.4 * sentiment
    final = "BUY" if final_score >= 0.55 else "HOLD" if final_score >= 0.45 else "SELL"
    out = {
        "ticker": ticker,
        "model_prob_up": model_prob,
        "news_sentiment": sentiment,
        "final_score": final_score,
        "final_decision": final,
        "llm": llm_summary,
    }
    print("Combined decision:", json.dumps(out, indent=2))
    return out


def run_all(config_path: str | Path = "config.yaml"):
    train_pipeline(config_path)
    evaluate(config_path)
    run_eda(config_path)
    live_inference(config_path)


def main():
    import argparse

    ap = argparse.ArgumentParser("dmp")
    ap.add_argument("command", choices=["train", "eval", "eda", "live", "all"]) 
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--loop", action="store_true", help="Continuously poll live inference")
    ap.add_argument("--interval", type=int, default=60, help="Seconds between live polls")
    args = ap.parse_args()

    if args.command == "train":
        train_pipeline(args.config)
    elif args.command == "eval":
        evaluate(args.config)
    elif args.command == "eda":
        run_eda(args.config)
    elif args.command == "live":
        if args.loop:
            import time
            print("Starting live loop. Ctrl+C to stop.")
            while True:
                try:
                    live_inference(args.config, args.ticker)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print("Error in live loop:", e)
                time.sleep(args.interval)
        else:
            live_inference(args.config, args.ticker)
    else:
        run_all(args.config)


if __name__ == "__main__":
    main()
