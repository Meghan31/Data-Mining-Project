from __future__ import annotations

import datetime as dt
from typing import List, Optional

import requests


class FinnhubClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://finnhub.io/api/v1"

    def _get(self, path: str, params: dict) -> dict:
        params = {**params, "token": self.api_key}
        r = requests.get(f"{self.base}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def quote(self, symbol: str) -> dict:
        return self._get("/quote", {"symbol": symbol})

    def company_news(self, symbol: str, start: str, end: str, max_items: int = 20) -> List[dict]:
        data = self._get("/company-news", {"symbol": symbol, "from": start, "to": end})
        return data[:max_items]

    def general_news(self, category: str = "general", min_id: Optional[int] = None) -> List[dict]:
        params = {"category": category}
        if min_id is not None:
            params["minId"] = min_id
        data = self._get("/news", params)
        return data

