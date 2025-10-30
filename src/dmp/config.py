from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class GeneralConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    figures_dir: str = "reports/figures"
    random_state: int = 42


class DataConfig(BaseModel):
    source: str = "yfinance"  # kaggle|yfinance
    tickers: List[str] = ["AAPL"]
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"


class FeaturesConfig(BaseModel):
    rsi_window: int = 14
    sma_windows: List[int] = [5, 10, 20]
    volatility_window: int = 10


class ModelConfig(BaseModel):
    type: str = "random_forest"
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: Optional[int] = None


class LiveConfig(BaseModel):
    provider: str = "finnhub"
    poll_seconds: int = 60
    default_ticker: str = "AAPL"


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_news: int = 10
    use_vader_fallback: bool = True


class AppConfig(BaseModel):
    general: GeneralConfig = GeneralConfig()
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    live: LiveConfig = LiveConfig()
    llm: LLMConfig = LLMConfig()

    @staticmethod
    def load(config_path: str | Path) -> "AppConfig":
        with open(config_path, "r") as f:
            content = yaml.safe_load(f)
        # Normalize date fields that YAML may parse as date objects
        if "data" in content:
            for k in ("start_date", "end_date"):
                v = content["data"].get(k)
                if v is not None and not isinstance(v, str):
                    try:
                        content["data"][k] = str(v)
                    except Exception:
                        pass
        cfg = AppConfig(**content)

        # Expand and create directories if needed
        artifacts = Path(cfg.general.artifacts_dir)
        figures = Path(cfg.general.figures_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)

        return cfg


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    return val


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
