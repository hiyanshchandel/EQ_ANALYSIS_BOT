"""Test doubles and factories. Fakes let the real graph run with no network or API keys."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.market_data import MarketDataError
from eq_analyst.agent.prompts import PromptRegistry
from eq_analyst.domain.models import (
    Fundamentals,
    NewsFeed,
    NewsItem,
    TechnicalSnapshot,
    TickerMatch,
    TickerSearchResult,
)


def make_history(days: int = 260, start: float = 100.0, daily_drift: float = 0.002) -> pd.DataFrame:
    """Synthetic daily candles with a steady drift and a little deterministic noise."""
    rng = np.random.default_rng(seed=42)
    close = start * np.cumprod(1 + daily_drift + rng.normal(0, 0.005, days))
    index = pd.bdate_range(end="2026-10-23", periods=days)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(days, 1_000_000.0),
        },
        index=index,
    )


def make_snapshot(**overrides: Any) -> TechnicalSnapshot:
    """A healthy uptrend by default; override fields to create risky situations."""
    values: dict[str, Any] = {
        "symbol": "TCS.NS",
        "currency": "INR",
        "as_of": date(2026, 10, 23),
        "price": 100.0,
        "change_1d_pct": 0.5,
        "sma_20": 98.0,
        "sma_50": 95.0,
        "sma_200": 90.0,
        "rsi_14": 55.0,
        "macd": 1.2,
        "macd_signal": 1.0,
        "macd_hist": 0.2,
        "bb_upper": 104.0,
        "bb_lower": 94.0,
        "atr_14": 2.0,
        "atr_pct": 2.0,
        "volume_ratio": 1.1,
        "high_52w": 110.0,
        "low_52w": 80.0,
    }
    values.update(overrides)
    return TechnicalSnapshot(**values)


TCS = TickerMatch(
    symbol="TCS.NS",
    name="Tata Consultancy Services Limited",
    exchange="NSI",
    exchange_display="NSE",
    quote_type="EQUITY",
)


class FakeMarketData:
    def __init__(
        self,
        matches: list[TickerMatch] | None = None,
        technical: TechnicalSnapshot | None = None,
        technical_error: bool = False,
        news: list[NewsItem] | None = None,
        debt_to_equity: float = 0.1,
    ) -> None:
        self.matches = [TCS] if matches is None else matches
        self.technical = technical or make_snapshot()
        self.technical_error = technical_error
        self.news = news if news is not None else [NewsItem(title="TCS wins a large deal")]
        self.debt_to_equity = debt_to_equity

    async def search_companies(self, query: str) -> TickerSearchResult:
        return TickerSearchResult(query=query, matches=self.matches)

    async def get_technical_snapshot(self, symbol: str) -> TechnicalSnapshot:
        if self.technical_error:
            raise MarketDataError(f"No price history found for symbol '{symbol}'.")
        return self.technical

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        return Fundamentals(
            symbol=symbol, name="TCS", pe_ratio=28.0, debt_to_equity=self.debt_to_equity
        )

    async def get_news(self, symbol: str, limit: int = 8) -> NewsFeed:
        return NewsFeed(symbol=symbol, items=self.news)


class FakeLLM:
    """Mimics `llm.with_structured_output(Schema).ainvoke(...)` with canned answers.

    `responses` maps a schema class to one answer, or a list of answers used in order.
    """

    def __init__(self, responses: dict[type[BaseModel], BaseModel | list[BaseModel]]) -> None:
        self.responses = responses
        self.calls: list[tuple[type[BaseModel], list[Any]]] = []

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredRunner:
        return _StructuredRunner(self, schema)


class _StructuredRunner:
    def __init__(self, llm: FakeLLM, schema: type[BaseModel]) -> None:
        self.llm, self.schema = llm, schema

    async def ainvoke(self, messages: list[Any], config: Any = None) -> BaseModel:
        self.llm.calls.append((self.schema, messages))
        answer = self.llm.responses[self.schema]
        return answer.pop(0) if isinstance(answer, list) else answer


class FakeFollowUpAgent:
    def __init__(self, answer: str = "Because momentum is strong.") -> None:
        self.answer = answer
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.inputs.append(payload)
        return {"messages": [*payload["messages"], AIMessage(self.answer)]}


def make_context(llm: FakeLLM, market_data: FakeMarketData | None = None) -> AgentContext:
    return AgentContext(
        llm=llm,  # type: ignore[arg-type]
        market_data=market_data or FakeMarketData(),
        prompts=PromptRegistry(langfuse=None),
        follow_up_agent=FakeFollowUpAgent(),  # type: ignore[arg-type]
        preferred_exchanges=("NSI", "BSE"),
    )
