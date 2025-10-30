from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dmp.config import AppConfig


def parse_env_file(path: str | Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def load_user_env(path: str | Path = "user_config.env") -> Dict[str, str]:
    env = parse_env_file(path)
    # Export to process environment so downstream libs can read them
    for k, v in env.items():
        if v != "":
            os.environ[k] = v
    return env


def apply_overrides(cfg: AppConfig) -> AppConfig:
    model_type = os.getenv("MODEL_TYPE")
    if model_type:
        cfg.model.type = model_type
    llm_model = os.getenv("LLM_MODEL")
    if llm_model:
        cfg.llm.model = llm_model
    ticker = os.getenv("TICKER")
    if ticker:
        cfg.live.default_ticker = ticker
        if ticker not in cfg.data.tickers:
            cfg.data.tickers = [ticker] + [t for t in cfg.data.tickers if t != ticker]
    return cfg

