# PPT Helper: Slide-by-Slide Guide

Use this guide to quickly build two presentations from the project outputs.

Important: Generate figures and metrics first by running `./up` (or `python -m dmp.cli all`). Figures will appear under `reports/figures/` and model metrics in `artifacts/`.

Contents
- Part A: Checkpoint Deck (Milestone/Interim) — 10–14 slides
- Part B: Final Presentation Deck — 18–28 slides
- Asset Map — where to find images/metrics to drop into slides

---

Part A — Checkpoint Deck
1) Title & Team
   - Title: "Real-Time Stock Analysis via ML + LLM News Fusion".
   - Subtitle: Goals, team members, date.

2) Problem Motivation
   - Pain points: stock data volume, noise, difficulty of interpretation.
   - One-liner objective: "Augment short-term movement prediction by fusing technical ML patterns with news context."

3) Data Sources
   - Historical: yfinance (or Kaggle alternative).
   - Live: Finnhub for quotes/news.
   - LLM: OpenAI for summarization (VADER fallback).
   - Include a small architecture icon (use diagram in this file or add a simple block diagram).

4) Pipeline Overview
   - Stages: Ingest → Feature Engineering → Model → Evaluation → Live + LLM → Aggregation.
   - Diagram: use simple blocks and arrows.

5) Features (Technical Indicators)
   - Daily returns, SMA ratios (5/10/20), RSI(14), rolling volatility(10).
   - What each feature captures; add simple bullet definitions.

6) EDA: Price Trends
   - Insert 2–3 price plots: `reports/figures/AAPL_close.png`, `.../MSFT_close.png`.
   - Note seasonality/events if visible.

7) EDA: Return Distribution
   - Insert `reports/figures/return_distribution.png`.
   - Comment on skew, fat tails; justification for robust models.

8) EDA: Feature Correlation
   - Insert `reports/figures/feature_correlation.png`.
   - Observations: SMA ratios correlate; RSI complements trend.

9) Baseline Model + Early Metrics
   - From `artifacts/metrics.json` and/or `artifacts/eval_metrics.json`.
   - Report Accuracy/Precision/Recall/F1. Add 2–3 interpretation bullets.

10) Next Milestones
   - Hyperparameter tuning, feature expansion, deeper news fusion.
   - Live loop polish and dashboard option.

---

Part B — Final Presentation Deck
1) Cover Slide
   - Title, team, timeframe.

2) Executive Summary
   - One slide with 3 bullets: what we built, how it works, key results.

3) Problem & Motivation
   - Business pain point and ML opportunity.

4) Related Work / Approaches
   - Moving-average crossovers, momentum signals, sentiment models.

5) Data & Collection
   - Historical (yfinance/Kaggle); Live (Finnhub); News (Finnhub + OpenAI).
   - Data window, tickers used.

6) Feature Engineering
   - Returns, SMA ratios (5/10/20), RSI(14), Volatility(10).
   - Equations or definitions; add a small table of windows.

7) Modeling Strategy
   - RandomForest and LogisticRegression; why tree ensemble for nonlinearity + robustness.
   - Train/test split; class balance notes.

8) Evaluation Metrics
   - Accuracy, Precision, Recall, F1; why they matter for up/down day capture.

9) Results: Offline
   - Insert numbers from `artifacts/eval_metrics.json` (accuracy, precision, recall, F1).
   - Confusion matrix discussion (from the JSON list or via a plotted matrix if you create one).

10) Error Analysis
   - When does the model fail? Sideways markets, news shocks, earnings surprise days.

11) Live Inference + News Fusion
   - Diagram: model prob + LLM sentiment → weighted final score.
   - Summarize aggregation weights (0.6 model, 0.4 sentiment by default).

12) LLM Summarization Examples
   - Screenshots or text snippets of summaries from OpenAI or VADER fallback.

13) End-to-End Demo Snapshot
   - Output example from a live run (BUY/HOLD/SELL decision with numbers).

14) System Architecture
   - Blocks: yfinance/Finnhub → features → model → eval → live → LLM → aggregator → output.

15) MLOps/Automation
   - `./up` creates venv, installs deps, runs pipeline, optionally starts live loop.
   - `./down` stops live loop; shows clean-up commands.

16) Limitations
   - Non-stationarity, regime changes, limited news coverage, latency.

17) Future Work
   - Feature store, sector conditioning, transformer-based time series, richer news embeddings, risk constraints.

18) Conclusion
   - Learnings; whether patterns aided short-term movement classification; next steps.

19) Appendix (optional)
   - Full metrics dump, feature importances, ablation comparisons.

---

Asset Map
- Figures: `reports/figures/`
  - `AAPL_close.png`, `MSFT_close.png`, `GOOGL_close.png`
  - `return_distribution.png`
  - `feature_correlation.png`
- Metrics: `artifacts/metrics.json`, `artifacts/eval_metrics.json`
- Live log (if loop enabled): `artifacts/live.log`

Copy Blocks (Optional)
- Combined decision rule: `final_score = 0.6 * model_prob_up + 0.4 * news_sentiment` → BUY/HOLD/SELL thresholds: 0.55/0.45.
- Feature rationale: SMA ratios capture trend strength; RSI flags overbought/oversold; volatility indicates uncertainty; returns provide momentum.

Tip: Keep slides clean. Use one key insight per chart. Include axes labels and date ranges.

