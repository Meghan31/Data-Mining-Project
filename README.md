# Data Mining Project
End-to-end stock analysis pipeline combining ML on historical data, real-time prices, and news sentiment via LLMs.

**Features**
- Data ingest from yfinance
- Feature engineering (returns, SMAs, RSI, volatility)
- Train classifier (logistic regression or random forest)
- Metrics: accuracy, precision, recall, F1
- EDA figures saved under `reports/figures`
- Live inference with Finnhub + news summarization via OpenAI
- CLI entrypoint `python -m dmp.cli <command>`

**Quickstart**
1) Create a virtual env and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2) Optional keys (export if you have them):
   - `export FINNHUB_API_KEY=...` (for live quotes/news)
   - `export OPENAI_API_KEY=...` (for LLM summarization)
   - `export KAGGLE_USERNAME=...` and `export KAGGLE_KEY=...` (if using Kaggle)
3) Run end-to-end:
   - `python -m dmp.cli all`  or `bash run.sh`

**Hands‑Off Usage**
- Edit only `user_config.env` to set API keys and choose models.
- `./up` runs everything (train, eval, EDA, and one‑shot live inference; starts a live loop if `LIVE_LOOP=true`).
- `./down` stops the live loop. It also prints cleanup commands for artifacts/venv.

**Commands**
- `train`: trains the model and saves artifacts in `artifacts/`
- `eval`: evaluates the saved model on the full frame
- `eda`: generates plots to `reports/figures`
- `live`: runs one live prediction + aggregated decision
- `all`: runs everything above

**Configuration**
- See `config.yaml` for tickers, dates, feature windows, and model settings.

**Notes**
- If Finnhub/OpenAI keys are missing, the pipeline still runs. It uses yfinance for data and VADER for sentiment fallback.
- This project is for educational purposes and not financial advice.
